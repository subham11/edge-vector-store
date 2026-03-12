// ──────────────────────────────────────────────────────────────
//  Android JNI entry point — installs the JSI HostObject
// ──────────────────────────────────────────────────────────────
#include <jni.h>
#include <jsi/jsi.h>
#include "../../cpp/bridge/EdgeStoreModule.h"

extern "C" JNIEXPORT void JNICALL
Java_com_sukshm_edgevectorstore_EdgeVectorStoreModule_nativeInstall(
    JNIEnv* /*env*/, jclass /*clazz*/, jlong jsiRuntimePtr) {
    auto* rt = reinterpret_cast<facebook::jsi::Runtime*>(jsiRuntimePtr);
    evs::installEdgeStoreModule(*rt);
}
