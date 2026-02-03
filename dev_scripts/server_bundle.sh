#!/bin/bash

# Script to create a git branch, stage all files, commit, and create a bundle

cd /root/lda

# Prompt for branch name
read -p "Enter branch name: " BRANCH_NAME

# Create and switch to new branch
git checkout -b "$BRANCH_NAME"

# Stage all files
git add -A

# Prompt for commit message
read -p "Enter commit message: " COMMIT_MSG

# Commit changes
git commit -m "$COMMIT_MSG"

# Create bundle
BUNDLE_PATH="/root/lda/${BRANCH_NAME}.bundle"
git bundle create "$BUNDLE_PATH" main.."$BRANCH_NAME"

echo "Bundle created at: $BUNDLE_PATH"
