# Caffe2 AICamera tutorial example

This is an example for using Caffe2 on Android. See [the original README](README.orig.md) for details.

## Using caffe2 from PyTorch master

PyTorch folder is at `$PYTORCH_ROOT`
This repository folder is at `$AICAMERA_ROOT`
Android NDK folder is at `$ANDROID_NDK`

Then, do the following:

1. Build caffe2 android libs and copy them over into AICamera app folder

```
# make sure $PYTORCH_ROOT, $AICAMERA_ROOT and $ANDROID_NDK are set
pushd $PYTORCH_ROOT

./scripts/build_android.sh
mv build_android build_android_arm

# copy headers
cp -r install/include/* $AICAMERA_ROOT/app/src/main/cpp/

# copy arm libs
rm -rf $AICAMERA_ROOT/app/src/main/jniLibs/armeabi-v7a/
mkdir $AICAMERA_ROOT/app/src/main/jniLibs/armeabi-v7a
cp -r build_android_arm/lib/lib* $AICAMERA_ROOT/app/src/main/jniLibs/armeabi-v7a/


./scripts/build_android.sh -DANDROID_ABI=x86
mv build_android build_android_x86

# copy x86 libs
rm -rf $AICAMERA_ROOT/app/src/main/jniLibs/x86/
mkdir $AICAMERA_ROOT/app/src/main/jniLibs/x86
cp -r build_android_x86/lib/lib* $AICAMERA_ROOT/app/src/main/jniLibs/x86/

```

2. Build the AICamera app using the `Build -> Make Project` menu option in Android Studio
