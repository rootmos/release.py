import argparse
import os
import pickle
import re

import git
import github
import semver

whoami = "release"
def env(var, default=None):
    return os.environ.get("RELEASE_" + var, default)

import logging
logger = logging.getLogger(whoami)

def figure_out_defaults():
    try:
        local = git.Repo(".", search_parent_directories=True)
        origin = local.remotes["origin"]
        return local.working_dir, re.sub(r'^(https://github.com/|git@github.com:)(.*)\.git$', r'\2', origin.url)

    except IndexError | git.InvalidGitRepositoryError:
        return None, None

def parse_args():
    parser = argparse.ArgumentParser(
            description = "release tool",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--log", default=env("LOG_LEVEL", "WARN"), help="set log level")

    default_local, default_repo = figure_out_defaults()
    parser.add_argument("--github-repo", default=env("GITHUB_REPOSITORY", os.environ.get("GITHUB_REPOSITORY", default_repo)), help="owner/repo GitHub repository")
    parser.add_argument("--local-repo", default=env("LOCAL_REPOSITORY", os.environ.get("GITHUB_WORKSPACE", default_local)), help="local clone of the repository")

    parser.add_argument("rev", metavar="REV", default=os.environ.get("GITHUB_SHA", "HEAD"), nargs="?", help="revision to release")

    return parser.parse_args()

def setup_logger(level):
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    f = logging.Formatter(fmt="%(asctime)s:%(name)s:%(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
    ch.setFormatter(f)

    logger.addHandler(ch)

def pickle_expr(thing, f, force=False, cache_dir=None):
    path = os.path.join(cache_dir or ".", f"{thing}.pickle")
    if os.path.exists(path) and not force:
        logger.debug("reading %s from: %s", thing, path)
        with open(path, "rb") as f:
            return pickle.load(f)

    x = f()
    with open(path, "wb") as f:
        pickle.dump(x, f)
    return x

def parse_dot_version(f):
    pattern = re.compile(r"^VERSION_([A-Z]+)=(.*)$")
    kwargs = {}
    for l in f:
        m = pattern.match(l.strip())
        k = m[1].lower()
        try:
            kwargs[k] = int(m[2])
        except ValueError:
            kwargs[k] = m[2]
    return semver.Version(**kwargs)

class Context:
    def __init__(self, args):
        if args.local_repo is None:
            raise NotImplementedError("clone repo")
        self.repo = git.Repo(args.local_repo)

        self.target = self.repo.commit(args.rev)
        logger.debug("resolved target commit: %s -> %s", args.rev, self.target)

        def mk_github():
            token = os.environ.get("GITHUB_TOKEN")
            if token is not None:
                auth = github.Auth.Token(token)
            else:
                auth = None
            return github.Github(auth=auth)
        self._github = mk_github

        def mk_github_repo():
            return self.github.get_repo(args.github_repo)
        self._github_repo = mk_github_repo

        self._releases = None

    @property
    def github(self):
        if callable(self._github):
            self._github = self._github()
        return self._github

    @property
    def github_repo(self):
        if callable(self._github_repo):
            self._github_repo = self._github_repo()
        return self._github_repo

    def dot_version(self, commit):
        try:
            t = commit.tree[".version"].data_stream.read().decode()
        except KeyError:
            return None

        v = parse_dot_version(t.splitlines())
        logger.debug(".version in %s: %s", commit.hexsha, v)
        return v

    @property
    def releases(self):
        def f():
            logger.info("fetching releases from: %s", self.github_repo.full_name)
            rs = {}
            for r in self.github_repo.get_releases():
                if r.draft:
                    continue

                rs[r.tag_name] = {
                    "tag_name": r.tag_name,
                    "name": r.name,
                    "prerelease": r.prerelease,
                }
            return rs
        if self._releases is None:
            thing = f"releases.{self.github_repo.owner.login}.{self.github_repo.name}"
            self._releases = pickle_expr(thing, f)
        return self._releases

    def resolve_commits_to_releases(self):
        c2r = {}
        for tag_name, r in self.releases.items():
            c = self.repo.tag(tag_name).commit
            logger.debug("resolved tag: %s -> %s", tag_name, c.hexsha)
            r["commit"] = c
            c2r[c] = r
        return c2r

    def find_previous_release(self):
        c2r = self.resolve_commits_to_releases()
        ptr, r = self.target, c2r.get(self.target)
        try:
            while r is None:
                if len(ptr.parents) == 0:
                    logger.debug("root commit: %s", ptr)
                    break

                p = ptr.parents[0]
                logger.debug("traversing: %s^ -> %s", ptr.hexsha, p.hexsha)
                ptr = p
                r = c2r.get(ptr)
        except IndexError:
            assert(r == None)
        return r

def prepare(ctx):
    v1 = ctx.dot_version(ctx.target)
    if v1 is None:
        print(f"no .version in {ctx.target}: nothing to do")
        return

    r = ctx.find_previous_release()

    if r is None:
        return {
            "version": v1,
            "to": ctx.target.hexsha,
        }

    c0 = r["commit"]
    v0 = ctx.dot_version(c0)
    if v0 is None:
        raise NotImplementedError()

    if v1 == v0:
        v1 = v1.bump_prerelease("")
    elif v1 > v0:
        return {
            "previous_version": v0,
            "version": v1,
            "to": ctx.target.hexsha,
            "from": c0
        }

    raise ValueError(f"refusing to downgrade version: {v0} -> {v1}")

def main():
    args = parse_args()
    setup_logger(args.log.upper())
    logger.debug(f"args: {args}")

    ctx = Context(args)
    print(prepare(ctx))

if __name__ == "__main__":
    main()
