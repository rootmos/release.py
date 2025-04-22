#export RELEASE_LOG_LEVEL=DEBUG
export RELEASE_LOG_LEVEL=WARN

GITHUB_TOKEN=$(cat "$HOME/.github_access_token")
export GITHUB_TOKEN

refresh() {
    git fetch --tag
    rm -f .release*
}

go() {
    #release --cache-releases -p "$PWD/release.py#The script" && release -R -n "README.md"
    release --cache-releases -n "$PWD/release.py"
}
