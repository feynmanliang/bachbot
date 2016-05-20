#!/usr/bin/zsh

rm mla.krn

echo "!!! m1a.krn - feynman liang - AC bach
**kern  **kern  **dynam 
*staff2 *staff1 *staff1/2
*>[A,A,B,B]     *>[A,A,B,B]     *>[A,A,B,B]
*>norep[A,B]    *>norep[A,B]    *>norep[A,B]
*>A     *>A     *>A
*clefF4 *clefG2 *clefG2
*k[]    *k[]    *k[]
*C:     *C:     *C:
*M4/4   *M4/4   *M4/4
*met(c) *met(c) *met(c)
*MM80  *MM80  *MM80" > mla.krn

cat ./scratch/b5_0.8.txt >> mla.krn

echo "==   ==  ==
*-   *-  *-
EOF" >> mla.krn
