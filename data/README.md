# data/

Not tracked by git. The corpora belong to their authors and are not redistributed here. See
[docs/data.md](../docs/data.md) for sources, licences and the dev/test exclusion applied to the
silver corpus.

Expected layout:

```
raw/          barec_sent_{train,validation,test}.parquet   as published
              blind_sent.parquet                           official blind test (no labels)
processed/    proc_{raw,d3tok}_{train,validation,test,blind}.parquet   built by slra_ot.cli.preprocess
              folds.parquet                                            built by slra_ot.cli.build_folds
              silver_{raw,d3tok}.parquet                               built by slra_ot.cli.build_silver
external/
  barec10m/   BAREC-10M as distributed (raw/ and mr/ trees)
```

Set `SLRA_OT_DATA` to move this root elsewhere.
