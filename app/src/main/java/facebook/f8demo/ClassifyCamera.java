package facebook.f8demo;

import android.Manifest;
import android.app.ActionBar;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.util.Size;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;

public class ClassifyCamera extends AppCompatActivity {
    private static final String TAG = "F8DEMO";
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


    static {
        System.loadLibrary("native-lib");
    }

    public native String torchTransform(Bitmap bitmap, int h, int w, byte[] Y, byte[] U, byte[] V,
                                                  int yRowStride, int rowStride, int pixelStride);
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
                if (! cameraIsOpen) {
                    cameraIsOpen = true;
                    openCamera();
                }
            }
        });
/*
        final GestureDetector gestureDetector = new GestureDetector(this.getApplicationContext(),
                new GestureDetector.SimpleOnGestureListener(){
            @Override
            public boolean onDoubleTap(MotionEvent e) {
                return true;
            }

            @Override
            public void onLongPress(MotionEvent e) {
                super.onLongPress(e);

            }

            @Override
            public boolean onDoubleTapEvent(MotionEvent e) {
                return true;
            }

            @Override
            public boolean onDown(MotionEvent e) {
                return true;
            }
        });

        outputView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                return gestureDetector.onTouchEvent(event);
            }
        });
*/
        assert outputView!= null;
        //outputView.setSurfaceTextureListener(textureListener);
        tv = (TextView) findViewById(R.id.sample_text);

    }


    /*View.SurfaceTextureListener textureListener = new View.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            //open your camera here
            openCamera();
        }
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Transform you image captured size according to the surface width and height
        }
        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    }; */
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
                        if (processing) {
                            image.close();
                            return;
                        }
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

                        predictedClass = torchTransform(outputBitmap, h, w, Y, U, V, yRowStride, rowStride, pixelStride);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                tv.setText(predictedClass);
                                outputView.setImageBitmap(outputBitmap);
                                processing = false;
                                //openCamera();
                            }
                        });

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
                    Toast.makeText(ClassifyCamera.this, "Configuration change", Toast.LENGTH_SHORT).show();
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
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(ClassifyCamera.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
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
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(ClassifyCamera.this, "You can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        /*if (textureView.isAvailable()) {
            openCamera();
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
