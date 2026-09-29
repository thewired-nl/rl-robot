{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    (python313.withPackages (pyPkgs: with pyPkgs; let
      stable-baselines3 = (pyPkgs.stable-baselines3.overridePythonAttrs rec {
        version = "2.9.0";
        src = fetchFromGitHub {
          owner = "DLR-RM";
          repo = "stable-baselines3";
          tag = "v${version}";
          hash = "sha256-vKbILFjQuD2gAkl3J3RA/vEo5UYqWttJ99kZdlEsqkY=";
        };
      });
      sbx-rl = (buildPythonPackage rec {
        pname = "sbx-rl";
        version = "0.28.1";
        src = fetchFromGitHub
          {
            owner = "araffin";
            repo = "sbx";
            rev = "v${version}";
            sha256 = "sha256-zAQCT8AfT4cDj5QZRvxT3Myy6EFrjRAxz34O7Tnjbms=";
          };
        doCheck = false;
        pyproject = true;
        build-system = [
          setuptools
        ];
        dependencies = [
          jax
          flax
          optax
          tqdm
          rich
          tf-keras
          stable-baselines3
          (callPackage ./tfp-nightly.nix { })
        ];
      });
    in
    [
      # 3d simulation
      mujoco

      # benchmarking
      vprof

      # for notebooks
      jupyter
      notebook
      mediapy
      matplotlib
      pyqt6
      ipython

      # ml
      stable-baselines3
      sbx-rl
      # torchWithRocm
      torch
      gymnasium
      numpy
      scipy
      onnxscript

      tensorboard # for web UI
      tensorboardx # for pytorch integration

      # esp32
      (buildPythonPackage rec {
        pname = "esp_ppq";
        version = "1.3.11";
        src = fetchPypi {
          inherit pname version;
          hash = "sha256-Pt88A5ixP+eZVnYojNjryBsCbX9nBDtRMtSSOZwvpbU=";
        };
        doCheck = false;
        pyproject = true;
        build-system = [
          hatchling
          setuptools
        ];
        pythonRelaxDeps = [ "onnx" ];
        pythonRemoveDeps = [ "onnxsim-prebuilt" ];
        dependencies = [
          cryptography
          flatbuffers
          numpy
          onnx
          onnxruntime
          toml
          tqdm

          (buildPythonPackage {
            pname = "onnxsim";
            version = "0.7.3";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/12/ce/3bd161619ba6829d8059af3a99e2ca5f2feb2684415b91be72d825e86969/onnxsim-0.7.3-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
              sha256 = "sha256-iPQLDT9DTBm/BinBfrWqPI94gIWyAcaujNqdQbg+1+M=";
            };
            propagatedBuildInputs = [ onnx rich ];
            doCheck = false;
          })
        ];
      })
    ]))
  ];
}
