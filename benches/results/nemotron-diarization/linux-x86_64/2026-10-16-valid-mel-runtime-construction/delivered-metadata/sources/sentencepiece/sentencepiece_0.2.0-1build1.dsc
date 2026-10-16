-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA512

Format: 3.0 (quilt)
Source: sentencepiece
Binary: sentencepiece, libsentencepiece0, libsentencepiece-dev, python3-sentencepiece
Architecture: any
Version: 0.2.0-1build1
Maintainer: Debian Science Maintainers <debian-science-maintainers@lists.alioth.debian.org>
Uploaders:  TSUCHIYA Masatoshi <tsuchiya@namazu.org>, Kentaro Hayashi <kenhys@xdump.org>
Homepage: https://github.com/google/sentencepiece
Standards-Version: 4.6.2
Vcs-Browser: https://salsa.debian.org/science-team/sentencepiece
Vcs-Git: https://salsa.debian.org/science-team/sentencepiece.git
Testsuite: autopkgtest
Build-Depends: debhelper-compat (= 13), protobuf-compiler, libprotobuf-dev, dh-python, python3-all-dev, quilt, cmake, python3-setuptools
Package-List:
 libsentencepiece-dev deb libdevel optional arch=any
 libsentencepiece0 deb libs optional arch=any
 python3-sentencepiece deb python optional arch=any
 sentencepiece deb science optional arch=any
Checksums-Sha1:
 e79a593131083b4ce2a77f11c977fd44c9d5337b 11988325 sentencepiece_0.2.0.orig.tar.gz
 08d3213ba0ad2650b317d975dc486b729b18e67f 11352 sentencepiece_0.2.0-1build1.debian.tar.xz
Checksums-Sha256:
 d72556a0cb343d6ea23264ecdbc0ff04dfecd879be4d2d7a7026b9de1af2e4d1 11988325 sentencepiece_0.2.0.orig.tar.gz
 056076fa62c114b52deef3ad4ed8ab9b1ac07938a5991011c25220cd6850f780 11352 sentencepiece_0.2.0-1build1.debian.tar.xz
Files:
 870bed5d2b9b2b5b4818cb4ee6831d48 11988325 sentencepiece_0.2.0.orig.tar.gz
 2a55f4d8e3e5f280dc2eb70a6ca669b9 11352 sentencepiece_0.2.0-1build1.debian.tar.xz

-----BEGIN PGP SIGNATURE-----

iQIzBAEBCgAdFiEEJeP/LX9Gnb59DU5Qr8/sjmac4cIFAmYXsz8ACgkQr8/sjmac
4cLk6A/8DB57fYyhz1cDyhU+VpkJJ8BFottzEItDBKFPAhTRzGeLzZEXzoWgTdzF
7VS/ve06lULqVLPPofDsI3pRblzlzbJrl6ZQ9W4PcrmG9+7u7Cjf50TOe5kdP5o8
qN4iOF5dtQ2H8BRB4/PPiLL1UrZOtcuJ8lRg3fh1UqlLo8yppMWU4XK96XC/w79U
7D2hmG+kuiLOe8ZG91YewVKjoR4W+QNm+wHbaqi4VfQ8pSnV+DDSsxBF9p4gnTvj
nghVrI/+oXKdqu9HbFVh9nLZmkoOHgWnOAC+lyAN/1u6qWkRUM9y7DGvE7gSVyPi
E2APtYq4BQ7QRD13PrYA2HPytQoGoMhf2PpNTJWh7M5nok/Rd6QkWJN4lO6fN2ff
liy2COQTqdnMVa050xVIpth1axEGr8i0g2nLbCIk+fBxlWdWNXN6YlfsgTUzGYyL
ZgmotrFwJmwbDJpL0u29+T+u1FE0Y8vQ93MnF2XG3i57AOSJrSCrP7yuxWa8Xq6S
Dx4iOl9Dp48WZgMUd7wtiueMZXSC1OAuyibyUelOAnYQMoWSVuQOgRTzWEPKnvYy
6vh5WhIhdw2/RrgTSy9Ol3vpPiVNwCwQrnUPUduG4sBo3GOawucyOxGzOk0HM2jx
/3fuZmoZ6c8ThNjwss6n1+gVH1hnqMdzsNKbv0mMBjwdsfur0GM=
=bOV5
-----END PGP SIGNATURE-----
