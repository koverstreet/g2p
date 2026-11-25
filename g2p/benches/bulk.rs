use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use g2p::bulk::{mul_add_matrix, mul_matrix, set_isa, Isa, MulTable};
use g2p::{g2p, GaloisField8};
use reed_solomon_erasure::galois_8::ReedSolomon;

g2p!(GF256, 8, modulus: 0b1_0001_1101);

const LEN: usize = 1 << 20;

fn tables(n: usize) -> Vec<MulTable> {
    (0..n)
        .map(|j| GF256::from(1 + j as u8).mul_table())
        .collect()
}

/// One data block folded into k parity blocks, and 10 encoded into k, by each ISA the CPU has -
/// and reed-solomon-erasure's encoding beside them. The bytes are the data's.
fn bulk(c: &mut Criterion) {
    let data: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8 * 37 + 1; LEN]).collect();
    let d: Vec<&[u8]> = data.iter().map(|d| &d[..]).collect();

    let mut fold = c.benchmark_group("fold");
    fold.throughput(Throughput::Bytes(LEN as u64));
    for k in [1, 2, 3, 6] {
        let (mut parity, m) = (vec![vec![0; LEN]; k], tables(k));
        for isa in Isa::all() {
            // SAFETY: Isa::all() found the CPU has it
            unsafe { set_isa(isa) };
            fold.bench_with_input(BenchmarkId::new(format!("{:?}", isa), k), &k, |b, _| {
                let mut p: Vec<&mut [u8]> = parity.iter_mut().map(|p| &mut p[..]).collect();
                b.iter(|| mul_add_matrix(&mut p, &d[..1], &m))
            });
        }
    }
    fold.finish();

    let mut encode = c.benchmark_group("encode");
    encode.throughput(Throughput::Bytes((d.len() * LEN) as u64));
    for k in [2, 4] {
        let (mut parity, m) = (vec![vec![0; LEN]; k], tables(k * d.len()));
        for isa in Isa::all() {
            // SAFETY: as above
            unsafe { set_isa(isa) };
            encode.bench_with_input(BenchmarkId::new(format!("{:?}", isa), k), &k, |b, _| {
                let mut p: Vec<&mut [u8]> = parity.iter_mut().map(|p| &mut p[..]).collect();
                b.iter(|| mul_matrix(&mut p, &d, &m))
            });
        }

        let rs = ReedSolomon::new(d.len(), k).expect("a code");
        let mut shards: Vec<Vec<u8>> = data.iter().cloned().chain(parity).collect();
        encode.bench_with_input(BenchmarkId::new("reed_solomon_erasure", k), &k, |b, _| {
            b.iter(|| rs.encode(&mut shards).expect("encoded"))
        });
    }
    encode.finish();
}

criterion_group!(benches, bulk);
criterion_main!(benches);
