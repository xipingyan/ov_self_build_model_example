# README
Just test how to add runtime info to OPS.

# Example code of python

```
    for op in model.get_ops():
        if op.get_type_name() == "MatMul":
            # Add runtime info: DisableFP16Compression for MatMul
            op.rt_info['precise_0'] = ''
```