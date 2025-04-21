export RELEASE_LOG_LEVEL=DEBUG

GITHUB_TOKEN=$(cat "$HOME/.github_access_token")
export GITHUB_TOKEN

refresh() {
    git fetch --tag
    rm ./*.pickle
}

go() {
    release --cache-releases -p "$PWD/release.py#The script" && release -R -n "README.md"
}
