# L2C-RD Software Pack

C implementations of the 32-bit, 64-bit, and 128-bit L2C-RD variants.

Each folder contains a reference program (`*-tweak.c`) and a correctness and performance program (`*-tweak-optimize.c`):

- `L2C-RD-32/`
- `L2C-RD-64/`
- `L2C-RD-128/` (128-bit block, 256-bit key)

## Run

For example, from the repository root:

```sh
gcc -O2 L2C-RD-32/L2C-RD-32-tweak.c -o l2crd32
./l2crd32
```
