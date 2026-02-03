#!/bin/bash

# Script to pull git bundle from server, unpack, and push to remote

# Prompt for connection details
read -p "Enter server IP: " IP
read -p "Enter server port: " PORT
read -p "Enter branch name: " BRANCH_NAME

# Pull down the bundle
BUNDLE_FILE="${BRANCH_NAME}.bundle"
scp -P "$PORT" "root@${IP}:/root/lda/${BUNDLE_FILE}" .

# Switch to main and pull latest
git checkout main
git pull

# Unpack the bundle
git bundle verify "$BUNDLE_FILE"
git fetch "$BUNDLE_FILE" "$BRANCH_NAME":"$BRANCH_NAME"

# Switch to the branch
git checkout "$BRANCH_NAME"

# Push to remote
git push -u origin "$BRANCH_NAME"

# Clean up bundle file
rm "$BUNDLE_FILE"

echo "Branch $BRANCH_NAME has been pushed to origin"
