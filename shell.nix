{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.docker_28
    pkgs.xorg.xhost
    
    (pkgs.python3.withPackages (ps: [
      ps.requests
      ps.docker
      ps.pexpect
      ps.empy

      ps.distro
      ps.docopt
      ps.pytest
      ps.pyyaml
    ]))
  ];
}


