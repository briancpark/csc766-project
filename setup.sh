# export BAZEL_VERSION=0.13.1
# mkdir /bazel && \
#     cd /bazel && \
#     wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#     chmod +x bazel-*.sh && \
#     ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#     cd / && \
#     rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh



cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip && \
    unzip -q android-ndk-r15c-linux-x86_64.zip && \
    rm -f android-ndk-r15c-linux-x86_64.zip

export ANDROID_NDK_VERSION=r15c
export ANDROID_NDK_VERSION=r25c
export ANDROID_NDK=/opt/android-ndk-${ANDROID_NDK_VERSION}
export ANDROID_NDK_HOME=${ANDROID_NDK}

# add to PATH
export PATH=${PATH}:${ANDROID_NDK_HOME}