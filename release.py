import argparse
from contextlib import closing
from dataclasses import dataclass
import os
import pickle
import re
import sys
from typing import Any, Callable, Iterable, TextIO

import git
import github
import semver

whoami = "release"
def env(var, default=None):
    return os.environ.get("RELEASE_" + var, default)

import logging
logger = logging.getLogger(whoami)

TAG_PREFIX="releases/v"

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

    parser.add_argument("-p", "--prepare-only", action="store_true")

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

def parse_dot_version(f: Iterable[str]) -> semver.Version:
    pattern = re.compile(r"^VERSION_([A-Z]+)=(.*)$")
    kwargs = {}
    for l in f:
        m = pattern.match(l.strip())
        if m is None:
            raise RuntimeError("unable to parse .version: " + l)
        k = m[1].lower()

        try:
            kwargs[k] = int(m[2])
        except ValueError:
            kwargs[k] = m[2]
    return semver.Version(**kwargs)

def emit_dot_version(v: semver.Version, f: TextIO) -> None:
    f.write(f"VERSION_MAJOR={v.major}\n")
    f.write(f"VERSION_MINOR={v.minor}\n")
    f.write(f"VERSION_PATCH={v.patch}\n")
    if v.prerelease is not None:
        f.write(f"VERSION_PRERELEASE={v.prerelease}\n")

def once[Self, A](f: Callable[[Self],A]) -> property:
    x = None
    def g(self):
        nonlocal x
        if x is None:
            x = f(self)
        return x
    return property(g)

@dataclass
class Release:
    tag_name: str
    name: str | None
    version: semver.Version
    commit: git.Commit

    @property
    def prerelease(self):
        return bool(self.version.prerelease)

class Context:
    def __init__(self, args):
        if args.local_repo is None:
            raise NotImplementedError("clone repo")
        self.repo = git.Repo(args.local_repo)

        self.target = self.repo.commit(args.rev)
        logger.debug("resolved target commit: %s -> %s", args.rev, self.target)

        self._github_repo = args.github_repo

        self._close: list[Any] = [self.repo]

    def close(self):
        for c in self._close:
            c.close()

    @once
    def github(self):
        token = os.environ.get("GITHUB_TOKEN")
        if token is not None:
            auth = github.Auth.Token(token)
        else:
            auth = None
        g = github.Github(auth=auth)
        self._close.append(g)
        return g

    @once
    def github_repo(self):
        return self.github.get_repo(self._github_repo)

    def dot_version(self, commit) -> semver.Version | None:
        try:
            t = commit.tree[".version"].data_stream.read().decode()
        except KeyError:
            return None

        v = parse_dot_version(t.splitlines())
        logger.debug(".version in %s: %s", commit.hexsha, v)
        return v

    @once
    def releases(self):
        def f():
            logger.info("fetching releases from: %s", self.github_repo.full_name)
            rs = {}
            for raw in self.github_repo.get_releases():
                if raw.draft:
                    continue

                if not raw.tag_name.startswith(TAG_PREFIX):
                    logger.warning("release %s does not start with '%s': skipping", raw.tag_name, TAG_PREFIX)
                    continue

                version = semver.Version.parse(raw.tag_name.removeprefix(TAG_PREFIX))

                commit = self.repo.tag(raw.tag_name).commit
                logger.debug("resolved tag: %s -> %s", raw.tag_name, commit.hexsha)
                dot_version = self.dot_version(commit)
                if dot_version is None:
                    logger.warning("release %s does not have a .version file: skipping", raw.tag_name)
                    continue

                assert(version.major == dot_version.major)
                assert(version.minor == dot_version.minor)
                assert(version.patch == dot_version.patch)

                if dot_version.prerelease is None:
                    assert(version.prerelease is None)
                else:
                    assert(version.prerelease is not None)
                    assert(version.prerelease.startswith(dot_version.prerelease))
                    assert(raw.prerelease)

                r = Release(
                    tag_name = raw.tag_name,
                    name = raw.name,
                    version = version,
                    commit = commit,
                )

                logger.info("found release: %s", r)
                rs[r.tag_name] = r
            return rs
        thing = f"releases.{self.github_repo.owner.login}.{self.github_repo.name}"
        return pickle_expr(thing, f)

    def find_previous_release(self, prereleases=False):
        c2r = {}
        for _, r in self.releases.items():
            c2r[r.commit] = r

        ptr, r = self.target, c2r.get(self.target)
        try:
            while r is None or (bool(prereleases) != r.prerelease):
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
        logger.info("no .version in {%s}: nothing to do", ctx.target)
        return
    logger.debug("preparing .version: %s", repr(v1))

    r = ctx.find_previous_release(v1.prerelease is not None)

    if r is not None:
        v0 = r.version
        if (v1.major == v0.major and v1.minor == v0.minor and v1.patch == v0.patch):
            if v1.prerelease is not None and v0.prerelease is not None:
                if v0.prerelease.startswith(v1.prerelease):
                    v1 = v0.bump_prerelease("")

    # make sure prereleases are "bumpable"
    if v1 == v1.bump_prerelease(""):
        if v1.prerelease:
            v1 = v1.replace(prerelease = v1.prerelease + ".1")
        else:
            v1 = v1.replace(prerelease = "1")

    if r is None:
        return {
            "version": v1,
            "to": ctx.target,
        }

    v0 = r.version
    if v1 > v0:
        return {
            "previous_version": v0,
            "version": v1,
            "to": ctx.target,
            "from": r.commit,
        }
    elif v1 == v0:
        raise ValueError(f"refusing to release same version: {v0} -> {v1}")
    else:
        raise ValueError(f"refusing to downgrade version: {v0} -> {v1}")

def main():
    args = parse_args()
    setup_logger(args.log.upper())
    logger.debug(f"args: {args}")

    with closing(Context(args)) as ctx:
        rel = prepare(ctx)
        if rel is None:
            return

        v1, to = rel["version"], rel["to"]
        v0, from_ = rel.get("previous_version"), rel.get("from")

        logger.info("version: %s -> %s", v0, v1)
        logger.info("commits: %s..%s", from_ or "", to)

        if args.prepare_only:
            emit_dot_version(v1, sys.stdout)
            return

        repo = ctx.github_repo
        base_url = f"https://github.com/{repo.full_name}"
        def commit_url(c):
            return base_url + "/commit/" + c.hexsha

        def compare_url(base, head):
            return base_url + "/compare/" + base.hexsha + ".." + head.hexsha

        msg = ""
        if from_ is None:
            if not bool(v1.prerelease):
                msg += "Initial release\n"
            else:
                msg += "Initial pre-release\n"
        elif from_ == to:
            msg += f"[{from_.hexsha[:7]}]({commit_url(from_)})\n"
        else:
            msg += f"[{from_.hexsha[:7]}..{to.hexsha[:7]}]({compare_url(from_, to)})\n"

        ctx.github_repo.create_git_tag_and_release(
            TAG_PREFIX + str(v1), # tag_name
            "", # tag_message
            str(v1), # release_name
            msg, # release_message
            to.hexsha,
            to.type,
            prerelease = bool(v1.prerelease)
        )

if __name__ == "__main__":
    main()
