/**
 * EVSExample — Edge Vector Store demo app
 *
 * Features:
 *   - RAG chat with 100K farmer records (Gemma 2B + vector search)
 *   - In-app benchmark harness
 */

import React from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import AppNavigator from './src/AppNavigator';

function App() {
  return (
    <SafeAreaProvider>
      <AppNavigator />
    </SafeAreaProvider>
  );
}

export default App;
