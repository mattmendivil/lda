#!/bin/bash

# Generate SSH host keys if they don't exist
[ ! -f /etc/ssh/ssh_host_rsa_key ] && ssh-keygen -A

# Start SSH daemon
/usr/sbin/sshd

# Keep container running
sleep infinity
