require "json"

pkg = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "EdgeVectorStore"
  s.version      = pkg["version"]
  s.summary      = pkg["description"]
  s.homepage     = pkg["homepage"] || "https://github.com/sukshm/edge-vector-store"
  s.license      = pkg["license"]
  s.author       = pkg["author"] || "sukshm"
  s.source       = { :git => ".git", :tag => s.version }

  s.ios.deployment_target = "13.0"

  # ── C++ sources ────────────────────────────────────────────
  s.source_files = [
    "cpp/**/*.{h,cpp}",
    "ios/**/*.{h,mm}",
    "third_party/usearch/c/*.{h,cpp}",
    "third_party/usearch/include/**/*.hpp",
    "third_party/miniz/*.{h,c}",
  ]

  # ── Include paths ──────────────────────────────────────────
  s.header_mappings_dir  = "."
  s.pod_target_xcconfig  = {
    "HEADER_SEARCH_PATHS" => [
      "\"$(PODS_TARGET_SRCROOT)/cpp\"",
      "\"$(PODS_TARGET_SRCROOT)/third_party/usearch/include\"",
      "\"$(PODS_TARGET_SRCROOT)/third_party/usearch/c\"",
      "\"$(PODS_TARGET_SRCROOT)/third_party/miniz\"",
    ].join(" "),
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
    "GCC_PREPROCESSOR_DEFINITIONS" => [
      "USEARCH_USE_SIMSIMD=0",
      "USEARCH_USE_FP16LIB=0",
      "USEARCH_USE_OPENMP=0",
    ].join(" "),
  }

  # React Native dependencies (auto-configures TurboModule/Codegen support)
  install_modules_dependencies(s)
end
