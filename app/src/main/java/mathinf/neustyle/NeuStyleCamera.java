package mathinf.neustyle;

import android.Manifest;
import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.util.Size;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.nio.ByteBuffer;
import java.util.Arrays;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;

public class NeuStyleCamera extends AppCompatActivity {
    private static final String TAG = "FULL_NEURAL_STYLE";
    private static final int REQUEST_CAMERA_PERMISSION = 200;

    private ImageView outputView;
    private String cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;
    private TextView tv;
    private String predictedClass = "none";
    private AssetManager mgr;
    private Bitmap outputBitmap;
    private boolean processing = false;
    private boolean cameraIsOpen = false;
    private Image image = null;
    private int sensorOrientation = 0;


    static {
        System.loadLibrary("native-lib");
    }

    public native String torchTransform(Bitmap bitmap, int h, int w, byte[] Y, byte[] U, byte[] V,
                                                  int yRowStride, int rowStride, int pixelStride, int sensorOrientation);
    public native void initCaffe2(AssetManager mgr);


    private class SetUpNeuralNetwork extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void[] v) {
            try {
                initCaffe2(mgr);
                predictedClass = "Neural net loaded! Inferring...";
            } catch (Exception e) {
                Log.d(TAG, "Couldn't load neural network.");
            }
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);

        ActivityManager manager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
        manager.getMemoryInfo(info);



        mgr = getResources().getAssets();

        new SetUpNeuralNetwork().execute();

        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        decorView.setSystemUiVisibility(uiOptions);

        setContentView(R.layout.activity_classify_camera);

        outputView = (ImageView) findViewById(R.id.outputView);
        outputView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        outputView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (! cameraIsOpen && ! processing) {
                    cameraIsOpen = true;
                    openCamera();
                }
            }
        });

        assert outputView!= null;
        //outputView.setSurfaceTextureListener(textureListener);
        tv = (TextView) findViewById(R.id.sample_text);

    }

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreview();
        }
        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }
        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };
    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }
    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    protected void createCameraPreview() {
        try {
            int width = 227;
            int height = 227;
            outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
            ImageReader reader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 4);
            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    try {

                        image = reader.acquireLatestImage();
                        if (image == null)
                            return;
                        if (processing) {
                            image.close();
                            return;
                        }
                        closeCamera();
                        processing = true;
                        int w = image.getWidth();
                        int h = image.getHeight();

                        ByteBuffer Ybuffer = image.getPlanes()[0].getBuffer();
                        ByteBuffer Ubuffer = image.getPlanes()[1].getBuffer();
                        ByteBuffer Vbuffer = image.getPlanes()[2].getBuffer();
                        // TODO: use these for proper image processing on different formats.
                        int yRowStride = image.getPlanes()[0].getRowStride();
                        int rowStride = image.getPlanes()[1].getRowStride();
                        int pixelStride = image.getPlanes()[1].getPixelStride();
                        byte[] Y = new byte[Ybuffer.capacity()];
                        byte[] U = new byte[Ubuffer.capacity()];
                        byte[] V = new byte[Vbuffer.capacity()];
                        Ybuffer.get(Y);
                        Ubuffer.get(U);
                        Vbuffer.get(V);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                tv.setText("Working...");
                            }
                        });
                        predictedClass = torchTransform(outputBitmap, h, w, Y, U, V, yRowStride, rowStride, pixelStride, sensorOrientation);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                tv.setText(predictedClass);
                                outputView.setImageBitmap(outputBitmap);
                                processing = false;
                            }
                        });
                    } catch (Exception e) {
                        tv.setText(e.getMessage());
                    } finally {
                        if (image != null) {
                            image.close();
                        }
                    }
                }
            };
            reader.setOnImageAvailableListener(readerListener, mBackgroundHandler);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            //captureRequestBuilder.addTarget(surface);
            captureRequestBuilder.addTarget(reader.getSurface());

            cameraDevice.createCaptureSession(Arrays.asList(reader.getSurface()), new CameraCaptureSession.StateCallback(){
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    if (null == cameraDevice) {
                        return;
                    }
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }
                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(NeuStyleCamera.this, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(NeuStyleCamera.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    protected void updatePreview() {
        if(null == cameraDevice) {
            return;
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
            cameraIsOpen = false;
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(NeuStyleCamera.this, "You can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        /* openCamera();*/
        /*if (textureView.isAvailable()) {

        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }*/
    }

    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }
}
