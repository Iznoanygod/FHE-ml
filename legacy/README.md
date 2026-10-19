# Legacy prototype (2022)

The original capstone code, written against PALISADE: a 784-200-10 MLP with
encrypted weights. The client decrypted the hidden layer, applied the sigmoid
in the clear and re-encrypted it. It is kept for reference only and is not
built. Known problems include: `Network::train` uses pre-activation values
where it needs post-activation ones, `dsigmoid` is applied to outputs that are
already sigmoids, `FHEMatrix::decrypt` returns an empty matrix, and the client
`fseek`s the wrong file.

The library in the repository root replaces it.
