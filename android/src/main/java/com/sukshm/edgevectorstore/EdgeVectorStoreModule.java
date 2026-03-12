package com.sukshm.edgevectorstore;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.module.annotations.ReactModule;

@ReactModule(name = EdgeVectorStoreModule.NAME)
public class EdgeVectorStoreModule extends ReactContextBaseJavaModule {
    static final String NAME = "EdgeVectorStore";

    static {
        System.loadLibrary("edge_vector_store");
    }

    public EdgeVectorStoreModule(ReactApplicationContext context) {
        super(context);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @ReactMethod
    public void install(Promise promise) {
        try {
            long jsiPtr = getReactApplicationContext()
                .getJavaScriptContextHolder()
                .get();
            nativeInstall(jsiPtr);
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject("ERR_INSTALL", "Failed to install JSI HostObject", e);
        }
    }

    private static native void nativeInstall(long jsiRuntimePtr);
}
