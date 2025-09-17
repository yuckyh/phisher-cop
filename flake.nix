# Use `nix develop` to enter the development environment
{
  description = "Development environment for the project";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        libraryPath = with pkgs;
          lib.makeLibraryPath [
            # I hate numpy and its dynamic linking >:(
            stdenv.cc.cc
            zlib
          ];
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [ uv ];
          shellHook = ''
            # Set up dynamic libraries for numpy
            export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libraryPath}"
          '';
        };
      });
}
