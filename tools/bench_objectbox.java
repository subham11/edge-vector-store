// ──────────────────────────────────────────────────────────────
//  ObjectBox Java Benchmark — HNSW on 100K farmer vectors
// ──────────────────────────────────────────────────────────────
//  Build & run:
//      cd tools && javac -cp "objectbox-java-4.0.3.jar" bench_objectbox.java
//      java -cp ".:objectbox-java-4.0.3.jar" bench_objectbox
//
//  Requires:
//      - ObjectBox Java JARs on classpath
//      - data/farmers_100k_vectors.bin  (100000 * 384 * 4 bytes)
//      - data/farmers_100k_queries.bin  (1000 * 384 * 4 bytes)
//      - data/farmers_100k_groundtruth.bin (1000 * 10 * 8 bytes)
// ──────────────────────────────────────────────────────────────

import io.objectbox.*;
import io.objectbox.annotation.*;
import io.objectbox.query.*;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;

// ── Entity ──────────────────────────────────────────────────

@Entity
class FarmerVector {
    @Id long id;

    @HnswIndex(dimensions = 384)
    float[] embedding;
}

// ── Benchmark ───────────────────────────────────────────────

public class bench_objectbox {

    static final int DIMS = 384;
    static final int NUM_VECTORS = 100_000;
    static final int NUM_QUERIES = 1_000;
    static final int TOP_K = 10;

    // Load raw float32 binary
    static float[] loadFloatBin(String path, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(count * 4);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        try (FileChannel ch = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            ch.read(buf);
        }
        buf.flip();
        float[] data = new float[count];
        buf.asFloatBuffer().get(data);
        return data;
    }

    // Load raw uint64 binary
    static long[] loadLongBin(String path, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(count * 8);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        try (FileChannel ch = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            ch.read(buf);
        }
        buf.flip();
        long[] data = new long[count];
        buf.asLongBuffer().get(data);
        return data;
    }

    static long dirSize(Path dir) throws IOException {
        long[] size = {0};
        Files.walk(dir).filter(Files::isRegularFile)
             .forEach(p -> { try { size[0] += Files.size(p); } catch (IOException e) {} });
        return size[0];
    }

    public static void main(String[] args) throws Exception {
        // Locate data directory
        String dataDir = null;
        for (String candidate : new String[]{"data", "../data", "../../data"}) {
            if (Files.exists(Path.of(candidate, "farmers_100k_vectors.bin"))) {
                dataDir = candidate;
                break;
            }
        }
        if (dataDir == null) {
            System.err.println("Cannot find data/ with vector files.");
            System.exit(1);
        }

        System.out.println("=== ObjectBox Java Benchmark ===");
        System.out.printf("Data: %s, Vectors: %d x %d%n", dataDir, NUM_VECTORS, DIMS);

        // Load data
        System.out.println("Loading vectors...");
        float[] allVectors = loadFloatBin(dataDir + "/farmers_100k_vectors.bin",
                                           NUM_VECTORS * DIMS);
        System.out.println("Loading queries...");
        float[] allQueries = loadFloatBin(dataDir + "/farmers_100k_queries.bin",
                                           NUM_QUERIES * DIMS);
        System.out.println("Loading ground-truth...");
        long[] groundTruth = loadLongBin(dataDir + "/farmers_100k_groundtruth.bin",
                                          NUM_QUERIES * TOP_K);

        // Temp directory for ObjectBox store
        Path tmpDir = Files.createTempDirectory("objectbox_bench");
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                Files.walk(tmpDir).sorted(Comparator.reverseOrder())
                     .map(Path::toFile).forEach(File::delete);
            } catch (IOException ignored) {}
        }));

        // Open store
        BoxStore store = MyObjectBox.builder()
                .directory(tmpDir.toFile())
                .build();
        Box<FarmerVector> box = store.boxFor(FarmerVector.class);

        // ── Insert ──
        System.out.println("\nInserting vectors...");
        long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long t0 = System.nanoTime();

        int BATCH = 1000;
        for (int start = 0; start < NUM_VECTORS; start += BATCH) {
            int end = Math.min(start + BATCH, NUM_VECTORS);
            List<FarmerVector> batch = new ArrayList<>(end - start);
            for (int i = start; i < end; i++) {
                FarmerVector fv = new FarmerVector();
                fv.embedding = Arrays.copyOfRange(allVectors, i * DIMS, (i + 1) * DIMS);
                batch.add(fv);
            }
            box.put(batch);
            if (((start / BATCH) + 1) % 20 == 0)
                System.out.printf("  Inserted %d/%d...%n", end, NUM_VECTORS);
        }

        long t1 = System.nanoTime();
        double insertSec = (t1 - t0) / 1e9;
        double insertThroughput = NUM_VECTORS / insertSec;
        long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        double memDeltaMB = (memAfter - memBefore) / 1e6;

        System.out.printf("  Insert: %.2fs (%.0f vec/s)%n", insertSec, insertThroughput);

        // Disk usage
        long diskBytes = dirSize(tmpDir);

        // ── Search ──
        System.out.println("\nSearching...");
        double[] latencies = new double[NUM_QUERIES];
        double recallSum = 0;

        for (int q = 0; q < NUM_QUERIES; q++) {
            float[] queryVec = Arrays.copyOfRange(allQueries, q * DIMS, (q + 1) * DIMS);

            long ts = System.nanoTime();
            Query<FarmerVector> query = box.query(
                FarmerVector_.embedding.nearestNeighbors(queryVec, TOP_K)
            ).build();
            List<FarmerVector> results = query.find();
            query.close();
            long te = System.nanoTime();

            latencies[q] = (te - ts) / 1e6; // ms

            // Recall
            int hits = 0;
            for (FarmerVector r : results) {
                long rid = r.id - 1; // ObjectBox IDs are 1-based
                for (int j = 0; j < TOP_K; j++) {
                    if (rid == groundTruth[q * TOP_K + j]) { hits++; break; }
                }
            }
            recallSum += (double) hits / TOP_K;
        }

        Arrays.sort(latencies);
        double searchMean = Arrays.stream(latencies).average().orElse(0);
        double searchP50  = latencies[latencies.length / 2];
        double searchP95  = latencies[(int)(latencies.length * 0.95)];
        double searchP99  = latencies[(int)(latencies.length * 0.99)];
        double recall     = recallSum / NUM_QUERIES;

        System.out.printf("  Search: mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms%n",
                          searchMean, searchP50, searchP95, searchP99);
        System.out.printf("  Recall@10: %.4f%n", recall);
        System.out.printf("  Disk: %.2f MB%n", diskBytes / 1e6);

        // ── Cold start ──
        store.close();
        long cs0 = System.nanoTime();
        BoxStore coldStore = MyObjectBox.builder()
                .directory(tmpDir.toFile())
                .build();
        // Run one query to fully warm
        Box<FarmerVector> coldBox = coldStore.boxFor(FarmerVector.class);
        float[] warmQuery = Arrays.copyOfRange(allQueries, 0, DIMS);
        Query<FarmerVector> warmQ = coldBox.query(
            FarmerVector_.embedding.nearestNeighbors(warmQuery, 1)
        ).build();
        warmQ.find();
        warmQ.close();
        long cs1 = System.nanoTime();
        double coldStartMs = (cs1 - cs0) / 1e6;
        coldStore.close();

        System.out.printf("  Cold start: %.2fms%n", coldStartMs);

        // ── Write JSON result ──
        String json = String.format(
            "{\"engine\":\"ObjectBox\",\"insertTimeSec\":%.3f,\"insertThroughput\":%.0f," +
            "\"indexBuildTimeSec\":0," +
            "\"searchLatencyMs\":{\"mean\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f}," +
            "\"recallAt10\":%.4f,\"diskUsageBytes\":%d,\"diskUsageMB\":%.2f," +
            "\"memoryDeltaMB\":%.2f,\"coldStartMs\":%.2f}",
            insertSec, insertThroughput,
            searchMean, searchP50, searchP95, searchP99,
            recall, diskBytes, diskBytes / 1e6,
            memDeltaMB, coldStartMs
        );

        String outPath = dataDir + "/bench_objectbox.json";
        Files.writeString(Path.of(outPath), json);
        System.out.printf("%nResults saved to %s%n", outPath);
    }
}
