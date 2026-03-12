// ──────────────────────────────────────────────────────────────
//  EdgeVectorStore — ObjC TurboModule for React Native 0.84+
//  Conforms to the codegen-generated NativeEdgeVectorStoreSpec
//  protocol and delegates to the C++ EdgeStore class.
// ──────────────────────────────────────────────────────────────
#ifdef __cplusplus
#import <EdgeVectorStoreSpec/EdgeVectorStoreSpec.h>
#import <ReactCommon/RCTTurboModule.h>

#include "cpp/core/EdgeStore.h"
#include "cpp/bridge/EdgeStoreModule.h"
#include "cpp/bridge/StoreRegistry.h"
#include "cpp/bench/BenchmarkEngine.h"
#include <memory>
#include <string>

// ── Subclass to intercept install() and inject JSI HostObject ──
namespace {
class EdgeVectorStoreSpecWithJSI
    : public facebook::react::NativeEdgeVectorStoreSpecJSI {
 public:
  EdgeVectorStoreSpecWithJSI(
      const facebook::react::ObjCTurboModule::InitParams &params)
      : NativeEdgeVectorStoreSpecJSI(params) {
    // Re-point the "install" entry so we can install the HostObject
    // before resolving the Promise back to JS.
    methodMap_["install"] = MethodMetadata{
        0, installWithHostObject};
  }

 private:
  static facebook::jsi::Value installWithHostObject(
      facebook::jsi::Runtime &rt,
      facebook::react::TurboModule &turboModule,
      const facebook::jsi::Value *args,
      size_t count) {
    evs::installEdgeStoreModule(rt);
    return static_cast<facebook::react::ObjCTurboModule &>(turboModule)
        .invokeObjCMethod(
            rt, facebook::react::PromiseKind, "install",
            @selector(install:reject:), args, count);
  }
};
} // anonymous namespace

@interface EdgeVectorStoreModule : NativeEdgeVectorStoreSpecBase <NativeEdgeVectorStoreSpec>
@end

@implementation EdgeVectorStoreModule {
    evs::StoreRegistry _registry;
}

RCT_EXPORT_MODULE(EdgeVectorStore)

+ (BOOL)requiresMainQueueSetup {
    return NO;
}

// ── TurboModule methods ──────────────────────────────────────

- (void)init:(NSString *)configJson
     resolve:(RCTPromiseResolveBlock)resolve
      reject:(RCTPromiseRejectBlock)reject {
    std::string json = [configJson UTF8String];
    if (!_registry.create(json)) {
        reject(@"ERR_INIT", @"EdgeVectorStore: native init failed", nil);
        return;
    }
    resolve(@(YES));
}

- (void)upsertDocuments:(NSString *)docsJson
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject {
    std::string json = [docsJson UTF8String];
    auto store = _registry.resolve(json);
    if (!store) { reject(@"ERR_NO_STORE", @"Store not initialised", nil); return; }
    if (!store->upsertDocuments(json)) {
        reject(@"ERR_UPSERT", @"upsertDocuments failed", nil);
        return;
    }
    resolve(nil);
}

- (void)upsertVectors:(NSString *)entriesJson
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject {
    std::string json = [entriesJson UTF8String];
    auto store = _registry.resolve(json);
    if (!store) { reject(@"ERR_NO_STORE", @"Store not initialised", nil); return; }
    if (!store->upsertVectors(json)) {
        reject(@"ERR_UPSERT", @"upsertVectors failed", nil);
        return;
    }
    resolve(nil);
}

- (void)search:(NSString *)optionsJson
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject {
    std::string json = [optionsJson UTF8String];
    auto store = _registry.resolve(json);
    if (!store) { reject(@"ERR_NO_STORE", @"Store not initialised", nil); return; }
    auto result = store->search(json);
    resolve([NSString stringWithUTF8String:result.c_str()]);
}

- (void)remove:(NSString *)idsJson
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject {
    std::string json = [idsJson UTF8String];
    auto store = _registry.resolve(json);
    if (!store) { reject(@"ERR_NO_STORE", @"Store not initialised", nil); return; }
    if (!store->remove(json)) {
        reject(@"ERR_REMOVE", @"remove failed", nil);
        return;
    }
    resolve(nil);
}

- (void)compact:(RCTPromiseResolveBlock)resolve
         reject:(RCTPromiseRejectBlock)reject {
    auto store = _registry.fallback();
    if (!store) { reject(@"ERR_NO_STORE", @"No store initialised", nil); return; }
    if (!store->compact()) {
        reject(@"ERR_COMPACT", @"compact failed", nil);
        return;
    }
    resolve(nil);
}

- (void)importPack:(NSString *)path
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject {
    auto store = _registry.fallback();
    if (!store) { reject(@"ERR_NO_STORE", @"No store initialised", nil); return; }
    if (!store->importPack([path UTF8String])) {
        reject(@"ERR_IMPORT", @"importPack failed", nil);
        return;
    }
    resolve(nil);
}

- (void)exportPack:(NSString *)path
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject {
    auto store = _registry.fallback();
    if (!store) { reject(@"ERR_NO_STORE", @"No store initialised", nil); return; }
    if (!store->exportPack([path UTF8String])) {
        reject(@"ERR_EXPORT", @"exportPack failed", nil);
        return;
    }
    resolve(nil);
}

- (void)getStats:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject {
    auto store = _registry.fallback();
    if (!store) { reject(@"ERR_NO_STORE", @"No store initialised", nil); return; }
    auto result = store->getStats();
    resolve([NSString stringWithUTF8String:result.c_str()]);
}

- (void)benchmarkRawANN:(NSString *)configJson
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject {
    std::string json = [configJson UTF8String];
    auto result = evs::benchmarkRawANN(json);
    if (result.find("error") != std::string::npos) {
        reject(@"ERR_BENCH", [NSString stringWithUTF8String:result.c_str()], nil);
        return;
    }
    resolve([NSString stringWithUTF8String:result.c_str()]);
}

- (void)install:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject {
    // No-op — JSI HostObject is installed automatically in getTurboModule:
    resolve(@(YES));
}

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params {
    return std::make_shared<EdgeVectorStoreSpecWithJSI>(params);
}

@end
#endif
