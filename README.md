This library exports an unordered, growable map type with heap-allocated contents, written as
`Registry<T>`. All contained values have their unique indices attributed
to them at the point of insertion. Indices are freed when the items are
removed.

Registries have *O*(1) indexing, *O*(1) removal and *O*(1) insertion.

# Examples

You can explicitly create a `Registry` with `Registry::new`:
```
let r: Registry<i32> = Registry::new();
```
You can [`insert`] values into the registry:
```
let mut r = Registry::new();
let index = r.insert(3);
```
Removing values works in much the same way:
```
let mut r = Registry::new();

let index = r.insert(3);
let three = r.remove(index);
```
Registries also support indexing (through the `Index` and `IndexMut` traits):
```
let mut r = Registry::new();
let i1 = r.insert(1);
let i2 = r.insert(4);

let one = r[i1];
r[i2] = r[i2] + 5;
```

The `Registry` type also exposes a full iterator API, so it can be used just like any other Rust container.
