import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Platform,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../AppNavigator';
import type { BenchResult } from '../services/BenchmarkRunner';

type Props = NativeStackScreenProps<RootStackParamList, 'Results'>;

const PHASE_NAMES: Record<number, string> = {
  1: 'Sanity (10K raw)',
  2: 'Core parity (100K raw)',
  3: 'Mobile reality (persist)',
  4: 'Compression (I8 vs F32)',
  5: 'Scale (250K)',
};

export default function ResultsScreen({ route }: Props) {
  let results: BenchResult[] = [];
  try {
    results = JSON.parse(route.params.results);
  } catch {
    // fallback to empty
  }

  const fmt = (n: number, decimals = 2) =>
    typeof n === 'number' ? n.toFixed(decimals) : '—';

  // Group by phase
  const byPhase = new Map<number, BenchResult[]>();
  for (const r of results) {
    const phase = r.phase ?? 0;
    if (!byPhase.has(phase)) byPhase.set(phase, []);
    byPhase.get(phase)!.push(r);
  }

  // Find max QPS per phase for relative bar
  const maxQPSByPhase = new Map<number, number>();
  for (const [phase, rows] of byPhase) {
    maxQPSByPhase.set(phase, Math.max(...rows.map((r) => r.qps || 0), 1));
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Benchmark Results</Text>

      {[...byPhase.entries()].map(([phase, rows]) => (
        <View key={phase}>
          <Text style={styles.phaseHeader}>
            Phase {phase}: {PHASE_NAMES[phase] ?? 'Unknown'}
          </Text>

          {rows.map((r, idx) => {
            const maxQ = maxQPSByPhase.get(phase)!;
            const barWidth = maxQ > 0 ? (r.qps / maxQ) * 100 : 0;

            return (
              <View key={`${phase}-${idx}`} style={styles.card}>
                <View style={styles.cardHeader}>
                  <Text style={styles.engineName}>{r.engine}</Text>
                  <Text style={styles.buildTag}>{r.buildConfig}</Text>
                </View>
                <Text style={styles.searchTag}>{r.searchConfig}</Text>

                <Text style={styles.subHeader}>Recall / Quality</Text>
                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Recall@10</Text>
                  <Text
                    style={[
                      styles.metricValue,
                      r.recallAt10 >= 0.90 && styles.good,
                      r.recallAt10 < 0.70 && styles.warn,
                    ]}
                  >
                    {fmt(r.recallAt10, 4)}
                  </Text>
                </View>
                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Dataset</Text>
                  <Text style={styles.metricValue}>
                    {(r.datasetSize / 1000).toFixed(0)}K × {r.dimensions}d
                  </Text>
                </View>

                <Text style={styles.subHeader}>Latency</Text>
                <View style={styles.latencyGrid}>
                  {(
                    [
                      ['mean', r.meanLatency],
                      ['p50', r.p50],
                      ['p95', r.p95],
                      ['p99', r.p99],
                    ] as [string, number][]
                  ).map(([label, val]) => (
                    <View key={label} style={styles.latencyCell}>
                      <Text style={styles.latencyLabel}>{label}</Text>
                      <Text style={styles.latencyValue}>{fmt(val, 3)}ms</Text>
                    </View>
                  ))}
                </View>

                <Text style={styles.subHeader}>Throughput</Text>
                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>QPS</Text>
                  <Text style={styles.metricValue}>{fmt(r.qps, 0)}</Text>
                </View>
                <View style={styles.barTrack}>
                  <View
                    style={[styles.barFill, { width: `${barWidth.toFixed(0)}%` }]}
                  />
                </View>
                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Insert</Text>
                  <Text style={styles.metricValue}>
                    {fmt(r.insertThroughput, 0)} vec/s
                  </Text>
                </View>
                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Build</Text>
                  <Text style={styles.metricValue}>
                    {fmt(r.buildTimeSec)}s
                  </Text>
                </View>

                <Text style={styles.subHeader}>Resources</Text>
                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Memory</Text>
                  <Text style={styles.metricValue}>
                    {fmt(r.memoryDeltaMB)}MB
                  </Text>
                </View>
                {r.diskUsageMB > 0 && (
                  <View style={styles.metricRow}>
                    <Text style={styles.metricLabel}>Disk</Text>
                    <Text style={styles.metricValue}>
                      {fmt(r.diskUsageMB)}MB
                    </Text>
                  </View>
                )}
                {r.coldStartMs > 0 && (
                  <View style={styles.metricRow}>
                    <Text style={styles.metricLabel}>Cold Start</Text>
                    <Text style={styles.metricValue}>
                      {fmt(r.coldStartMs)}ms
                    </Text>
                  </View>
                )}
                {r.warmStartMs > 0 && (
                  <View style={styles.metricRow}>
                    <Text style={styles.metricLabel}>Warm Start</Text>
                    <Text style={styles.metricValue}>
                      {fmt(r.warmStartMs)}ms
                    </Text>
                  </View>
                )}
              </View>
            );
          })}
        </View>
      ))}

      {results.length === 0 && (
        <Text style={styles.empty}>No results to display.</Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
    padding: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#e94560',
    marginBottom: 20,
  },
  phaseHeader: {
    fontSize: 16,
    fontWeight: '700',
    color: '#e94560',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginTop: 20,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e94560',
    paddingBottom: 6,
  },
  card: {
    backgroundColor: '#1a1a2e',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2,
  },
  engineName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#ffffff',
  },
  buildTag: {
    fontSize: 11,
    color: '#66bb6a',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
  },
  searchTag: {
    fontSize: 11,
    color: '#8888aa',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
    marginBottom: 8,
  },
  subHeader: {
    fontSize: 12,
    fontWeight: '600',
    color: '#e94560',
    marginTop: 10,
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a4e',
  },
  metricLabel: {
    color: '#8888aa',
    fontSize: 13,
  },
  metricValue: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '500',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
  },
  latencyGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 4,
  },
  latencyCell: {
    backgroundColor: '#0f0f23',
    borderRadius: 8,
    padding: 6,
    minWidth: 68,
    alignItems: 'center',
  },
  latencyLabel: {
    color: '#8888aa',
    fontSize: 10,
    textTransform: 'uppercase',
    marginBottom: 2,
  },
  latencyValue: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
  },
  barTrack: {
    height: 6,
    backgroundColor: '#0f0f23',
    borderRadius: 3,
    overflow: 'hidden',
    marginTop: 4,
    marginBottom: 4,
  },
  barFill: {
    height: '100%',
    backgroundColor: '#66bb6a',
    borderRadius: 3,
  },
  good: {
    color: '#66bb6a',
  },
  warn: {
    color: '#ffa726',
  },
  empty: {
    color: '#8888aa',
    textAlign: 'center',
    marginTop: 40,
    fontSize: 16,
  },
});
