import git
import re

try:
    local = git.Repo(".", search_parent_directories=True)

    print(local.git_dir)

    origin = local.remotes["origin"]

    ssh_url = origin.url
    public_url = "https://github.com/rootmos/release.py.git"

    print(re.sub(r'^(https://github.com/|git@github.com:)(.*)\.git$', r'\2', ssh_url))
    print(re.sub(r'^(https://github.com/|git@github.com:)(.*)\.git$', r'\2', public_url))

except IndexError | git.InvalidGitRepositoryError:
    pass
