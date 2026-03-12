# EVSExample — Edge Vector Store Demo

React Native bare CLI demo app for `@sukshm/edge-vector-store`.

## Features

- **Farmer Assistant**: RAG-powered chat over 100K Indian farmer records using Gemma 2 2B (via llama.cpp) + vector search
- **Benchmark**: In-app comparison of EdgeVectorStore vs brute-force search

## Setup

### 1. Install dependencies

```bash
cd example
npm install
cd ios && pod install && cd ..
```

### 2. Generate the 100K dataset (one-time)

```bash
# From the repo root (edge-vector-store/)
pip install sentence-transformers numpy tqdm
python tools/generate_dataset.py        # → data/farmers_100k.json + .npy
python tools/convert_vectors.py         # → data/*_vectors.bin, *_queries.bin, *_groundtruth.bin
```

### 3. (Optional) Download models for on-device inference

**Embedding model** (for query-time embedding):
```bash
# Download all-MiniLM-L6-v2 ONNX and place in app Documents dir
# ~22 MB — enables on-device query embedding
```

**LLM** (for conversational RAG):
```bash
# Download gemma-2-2b-it-Q4_K_M.gguf (~1.5 GB) and place in app Documents dir
# Without it, the app uses template-based responses
```

### 4. Run the C++ benchmark (desktop)

```bash
mkdir -p build/desktop && cd build/desktop
cmake ../.. -DEVS_DESKTOP_TEST=ON -DEVS_BENCHMARK=ON
make -j
./evs_bench    # Requires data/ from step 2
```

### 5. Run the app

```bash
npx react-native run-ios
# or
npx react-native run-android
```

## Architecture

```
example/
├── App.tsx                  # Entry point
├── src/
│   ├── AppNavigator.tsx     # React Navigation stack
│   ├── screens/
│   │   ├── HomeScreen.tsx   # Landing page
│   │   ├── ChatScreen.tsx   # RAG chat interface
│   │   ├── BenchmarkScreen.tsx  # Engine selection + run
│   │   └── ResultsScreen.tsx    # Benchmark results table
│   └── services/
│       ├── ChatService.ts       # RAG pipeline orchestrator
│       ├── VectorSearchService.ts # EdgeVectorStore wrapper
│       ├── QueryEmbedder.ts     # ONNX MiniLM embedding
│       ├── BenchmarkRunner.ts   # In-app benchmark harness
│       └── PromptTemplates.ts   # Gemma 2B prompt formatting
```

## Benchmarking

### In-app (JavaScript)
Uses 1K random vectors for quick comparison. Select engines and tap "Run Benchmark".

### C++ level (100K vectors)
```bash
./evs_bench   # Tests: Flat (brute-force), USearch (raw HNSW), EdgeVectorStore (full stack)
```

### LanceDB (Python)
```bash
pip install lancedb numpy pyarrow
python tools/bench_lancedb.py
```

### ObjectBox (Java)
```bash
javac -cp "objectbox-java.jar" tools/bench_objectbox.java
java -cp ".:objectbox-java.jar" bench_objectbox
```

### Compare all results
```bash
python tools/compare_results.py   # → data/benchmark_report.md
```

# Getting Started

> **Note**: Make sure you have completed the [Set Up Your Environment](https://reactnative.dev/docs/set-up-your-environment) guide before proceeding.

## Step 1: Start Metro

First, you will need to run **Metro**, the JavaScript build tool for React Native.

To start the Metro dev server, run the following command from the root of your React Native project:

```sh
# Using npm
npm start

# OR using Yarn
yarn start
```

## Step 2: Build and run your app

With Metro running, open a new terminal window/pane from the root of your React Native project, and use one of the following commands to build and run your Android or iOS app:

### Android

```sh
# Using npm
npm run android

# OR using Yarn
yarn android
```

### iOS

For iOS, remember to install CocoaPods dependencies (this only needs to be run on first clone or after updating native deps).

The first time you create a new project, run the Ruby bundler to install CocoaPods itself:

```sh
bundle install
```

Then, and every time you update your native dependencies, run:

```sh
bundle exec pod install
```

For more information, please visit [CocoaPods Getting Started guide](https://guides.cocoapods.org/using/getting-started.html).

```sh
# Using npm
npm run ios

# OR using Yarn
yarn ios
```

If everything is set up correctly, you should see your new app running in the Android Emulator, iOS Simulator, or your connected device.

This is one way to run your app — you can also build it directly from Android Studio or Xcode.

## Step 3: Modify your app

Now that you have successfully run the app, let's make changes!

Open `App.tsx` in your text editor of choice and make some changes. When you save, your app will automatically update and reflect these changes — this is powered by [Fast Refresh](https://reactnative.dev/docs/fast-refresh).

When you want to forcefully reload, for example to reset the state of your app, you can perform a full reload:

- **Android**: Press the <kbd>R</kbd> key twice or select **"Reload"** from the **Dev Menu**, accessed via <kbd>Ctrl</kbd> + <kbd>M</kbd> (Windows/Linux) or <kbd>Cmd ⌘</kbd> + <kbd>M</kbd> (macOS).
- **iOS**: Press <kbd>R</kbd> in iOS Simulator.

## Congratulations! :tada:

You've successfully run and modified your React Native App. :partying_face:

### Now what?

- If you want to add this new React Native code to an existing application, check out the [Integration guide](https://reactnative.dev/docs/integration-with-existing-apps).
- If you're curious to learn more about React Native, check out the [docs](https://reactnative.dev/docs/getting-started).

# Troubleshooting

If you're having issues getting the above steps to work, see the [Troubleshooting](https://reactnative.dev/docs/troubleshooting) page.

# Learn More

To learn more about React Native, take a look at the following resources:

- [React Native Website](https://reactnative.dev) - learn more about React Native.
- [Getting Started](https://reactnative.dev/docs/environment-setup) - an **overview** of React Native and how setup your environment.
- [Learn the Basics](https://reactnative.dev/docs/getting-started) - a **guided tour** of the React Native **basics**.
- [Blog](https://reactnative.dev/blog) - read the latest official React Native **Blog** posts.
- [`@facebook/react-native`](https://github.com/facebook/react-native) - the Open Source; GitHub **repository** for React Native.
