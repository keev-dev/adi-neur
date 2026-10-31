# Deploy Adi on Oracle Cloud Always Free

Target: one Ampere A1 VM, 2 OCPU, 12 GB RAM, Ubuntu 24.04 ARM.
Two independent simulations, one per core, ports 8000 and 8001.
The app binds to localhost. You reach it through an SSH tunnel.

## 1. Create the VM

In the Oracle Cloud console:

1. Compute -> Instances -> Create instance.
2. Image: Ubuntu 24.04 (aarch64).
3. Shape: VM.Standard.A1.Flex.
4. OCPUs: 2. Memory: 12 GB.
5. Add your SSH public key.
6. Boot volume: default is enough.
7. Create.

If you see "out of host capacity", change the availability domain or try later.
A1 capacity is often full.

Note the public IP.

## 2. Copy the project

From your machine:

```
ssh -i <key> ubuntu@<ip> "sudo mkdir -p /opt/adi && sudo chown ubuntu:ubuntu /opt/adi"
rsync -av --exclude data --exclude .git --exclude __pycache__ \
  ./ ubuntu@<ip>:/opt/adi/
```

Or use `scp -r`.

Do not copy `data/`. The VM starts its own runs.

## 3. Install

```
ssh -i <key> ubuntu@<ip>
cd /opt/adi
bash deploy/setup.sh
```

The script creates a venv, installs torch and the server, and enables two
systemd services: `adi@1` and `adi@2`.

## 4. Open it

On your machine:

```
ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 ubuntu@<ip>
```

Then open:

- http://127.0.0.1:8000
- http://127.0.0.1:8001

Port 8000 is seed 7. Port 8001 is seed 8.

## 5. Operate

```
systemctl status adi@1 adi@2
journalctl -u adi@1 -f
sudo systemctl restart adi@1
sudo systemctl stop adi@1 adi@2
```

Each instance writes to its own folder:

```
/opt/adi/data/runs/<timestamp>_s<seed>/
```

Pull the data back:

```
rsync -av ubuntu@<ip>:/opt/adi/data/runs/ ./data/oracle/
```

## 6. Size and limits

The Always Free A1 allowance is 1,500 OCPU hours and 9,000 GB hours per month.
Two cores running 24/7 use about 1,488 OCPU hours and 8,928 GB hours. That fits,
with very little margin. Do not add a third instance.

Torch on aarch64 uses the CPU wheel. No GPU is needed.

## 7. Security

The services bind to 127.0.0.1. Only SSH is open. Do not open port 8000 to the
internet. Use the tunnel.

If you want a public page, put it behind Nginx or Caddy with basic auth and TLS.
That is a separate step.

## 8. Troubleshooting

- `adi@1` fails: `journalctl -u adi@1 -n 100 --no-pager`
- torch install fails: check `python3 --version` and `uname -m`. It must be aarch64.
- WebSocket does not connect: the tunnel must forward both ports.
- The sim is slow: two instances share two cores. That is the design.

## 9. Safe updates

The VM should be a git clone of your repo. To update:

```
ssh -i <key> ubuntu@<ip> "cd /opt/adi && bash deploy/pull.sh --restart"
```

The script does this:

1. Pauses both instances.
2. Writes a checkpoint for each.
3. Pulls the code.
4. Installs dependencies.
5. Restarts the services.
6. Resumes both instances.

The checkpoint carries the population, the weights, the Adam state, the rooms,
the RNG, the focus, and the step. The run continues from the same point.

Autosave runs every 10 minutes. On SIGTERM the service saves before it exits.

Checkpoint files:

```
/opt/adi/data/checkpoints/instance_1.pt
/opt/adi/data/checkpoints/instance_2.pt
```

To update code without restarting the live run:

```
bash deploy/pull.sh
```

The new code is on disk. The next restart will use it and resume.

### First-time git setup on the VM

If the project was copied with rsync:

```
cd /opt/adi
git init
git remote add origin <repo-url>
git fetch origin
git reset --hard origin/main
```

The `data/` folder is in `.gitignore`. The reset does not remove it.

For a private repo, add a read-only deploy key to the VM and to the repo:

```
ssh-keygen -t ed25519 -f ~/.ssh/adi_deploy -N ""
cat ~/.ssh/adi_deploy.pub
```

Add that public key under GitHub -> Settings -> Deploy keys. Use the SSH remote:

```
git remote set-url origin git@github.com:<user>/<repo>.git
```

