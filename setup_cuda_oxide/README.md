# Setting up cuda-oxide.

[cuda-oxide](https://github.com/NVlabs/cuda-oxide) is an experimental rust to CUDA compiler from NVIDIA. To get set up to use the devcontainer, a node version manager and a node install are needed: [fnm](https://github.com/Schniz/fnm) is an option. Install `node` with `fnm install --lts`, and make sure it's on the path and do `npm install -g @devcontainers/cli`. You will also need docker. Try the [rootless](https://docs.docker.com/engine/security/rootless/) version. 

Docker config is in `~/.config/docker/daemon.json`, and the docker data is written to `~/.local/share/docker`. `cuda-oxide` dev container is approximately 16G after being built; it's useful to put all this somewhere other than in home directory. Symlinking or adding 
```json 
  {
    "data-root": "/path/to/docker-rootless-data"
  }
```

to `daemon.json` is the way to set where docker stores the data. 

`start` the docker service before running the `devcontainer`, and confirm the data directory and rootless status.   
```bash
  docker_start_rootless(){
    systemctl --user start docker.service
    docker info | grep "Root Dir"
    docker info | grep rootless
  } 
```

Pull the repo and start the container. 
```bash
   git clone https://github.com/NVlabs/cuda-oxide.git
   devcontainer up --workspace-folder . 
``` 

On first `devcontainer up` it will build the container. Once it's up 
```bash
    devcontainer exec --workspace-folder . cargo oxide doctor
```
kicks off the container. 

⚠️ Note on Permissions: It's possible that there are permission issues. If you are local system, rather than a server with many user, `chmod -R ugo+rw ./cuda-oxide/` is an quickfix, albeit insecure. 

To kick off with vecadd
```bash
    devcontainer exec --workspace-folder . cargo oxide run vecadd
```
Aliasing `devcontainer` with `alias cargo-cuda="devcontainer exec --workspace-folder . cargo"` improves the ergonomics. 

