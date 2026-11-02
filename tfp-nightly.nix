# Updated to 0.26 nightly build from https://github.com/NixOS/nixpkgs/blob/9bb52223f4f6a0cc606fd0852d8784f896c137d5/pkgs/development/python-modules/tensorflow-probability/default.nix 
{ lib
, stdenv
, buildBazelPackage
, fetchFromGitHub
, python
, setuptools
, wheel
, absl-py
, bazel
, cctools
, buildPythonPackage
, cloudpickle
, decorator
, dm-tree
, gast
, keras
, numpy
, six
, tensorflow
}:

let
  version = "0.26.0.dev20260924";
  pname = "tfp-nightly";

  # first build all binaries and generate setup.py using bazel
  bazel-wheel = buildBazelPackage {
    name = "tfp_nightly-${version}-py2.py3-none-any.whl";
    src = fetchFromGitHub {
      owner = "tensorflow";
      repo = "probability";
      rev = "d135a50f176224d654c62149be8cccc0886c0ed7";
      hash = "sha256-6WzjPpVvUq9QkfmQDWNV1CoqlpCaLhLQNKNseu2h0KM=";
    };

    nativeBuildInputs = [
      absl-py
      # needed to create the output wheel in installPhase
      python
      setuptools
      tensorflow
      wheel
    ];

    bazel = bazel;

    bazelTargets = [ ":pip_pkg" ];
    bazelFlags = [ "--noenable_bzlmod" ];
    removeRulesCC = false;
    LIBTOOL = lib.optionalString stdenv.hostPlatform.isDarwin "${cctools}/bin/libtool";

    fetchAttrs = {
      sha256 = "sha256-RzX8Shr8It9EahXDd3KotcZ2CjFcuzZVyK5aQutLPA4=";
    };

    buildAttrs = {
      preBuild = ''
        patchShebangs .
      '';

      installPhase = ''
        # work around timestamp issues
        # https://github.com/NixOS/nixpkgs/issues/270#issuecomment-467583872
        export SOURCE_DATE_EPOCH=315532800

        # First build, then move. Otherwise pip_pkg would create the dir $out
        # and then put the wheel in that directory. However we want $out to
        # point directly to the wheel file.
        # NOTE: removed --release to build tfp_nightly package
        ./bazel-bin/pip_pkg .
        mv *.whl "$out"
      '';
    };
  };
in
buildPythonPackage {
  inherit version pname;
  format = "wheel";

  src = bazel-wheel;

  dependencies = [
    cloudpickle
    decorator
    dm-tree
    gast
    keras
    numpy
    six
    tensorflow
  ];

  doCheck = false;

  # Ideally, we run unit tests with pytest, but in checkPhase, only the Bazel-build wheel is available.
  # But it seems not guaranteed that running the tests with pytest will even work, see
  # https://github.com/tensorflow/probability/blob/c2a10877feb2c4c06a4dc58281e69c37a11315b9/CONTRIBUTING.md?plain=1#L69
  # Ideally, tests would be run using Bazel. For now, lets's do a...

  # sanity check
  pythonImportsCheck = [ "tensorflow_probability" ];

  meta = {
    description = "Library for probabilistic reasoning and statistical analysis";
    homepage = "https://www.tensorflow.org/probability/";
    changelog = "https://github.com/tensorflow/probability/releases/tag/v${version}";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ GaetanLepage ];
  };
}
