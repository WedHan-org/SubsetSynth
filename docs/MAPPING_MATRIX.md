# Core Mapping Contract

## 9 learned primitive mappings

### 1 → 1

1. T1 → T2
2. T1 → FS
3. T2 → T1
4. T2 → FS
5. FS → T1
6. FS → T2

### 1+1 → 1

7. T1+T2 → FS
8. T1+FS → T2
9. T2+FS → T1

## 3 compositional configurations

10. T1 → {T2, FS}
11. T2 → {T1, FS}
12. FS → {T1, T2}

The final three reuse one source latent and issue two target queries. They are not separately encoded task identities.
