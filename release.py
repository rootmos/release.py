import argparse
from contextlib import closing
from dataclasses import asdict, astuple, dataclass
import json
import os
import pickle
import re
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

    parser.add_argument("--log", default=env("LOG_LEVEL", "INFO"), help="set log level")

    parser.add_argument("-n", "--dry-run", action="store_true")

    default_local, default_repo = figure_out_defaults()
    parser.add_argument("--github-repo", default=env("GITHUB_REPOSITORY", os.environ.get("GITHUB_REPOSITORY", default_repo)), help="owner/repo GitHub repository")
    parser.add_argument("--local-repo", default=env("LOCAL_REPOSITORY", os.environ.get("GITHUB_WORKSPACE", default_local)), help="local clone of the repository")
    parser.add_argument("--rev", metavar="REV", default=os.environ.get("GITHUB_SHA", "HEAD"), help="revision to release")

    parser.add_argument("--dot-release", default=".release", help="write a .version-formatted file with the release version (relative the local repository's workdir)")
    parser.add_argument("--dot-release-json", default=".release.json", help="write a json file with the (intended) release metadata (relative the local repository's workdir)")

    action = parser.add_mutually_exclusive_group()
    action.add_argument("-p", "--prepare", dest="action", action="store_const", const="prepare")
    action.add_argument("-r", "--release", dest="action", action="store_const", const="release")
    action.add_argument("-R", "--release-prep", dest="action", action="store_const", const="release-prep")

    args = parser.parse_args()

    if args.action is None:
        args.action = "release"

    return args

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

    def dot_version(self, commit: git.Commit) -> semver.Version | None:
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

@dataclass
class Range:
    to: git.Commit
    v1: semver.Version
    from_: git.Commit | None
    v0: semver.Version | None

def prepare_range(ctx) -> Range | None:
    to = ctx.target
    v1 = ctx.dot_version(to)
    if v1 is None:
        logger.info("no .version in {%s}: nothing to do", to)
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

    v0, from_ = None, None
    if r is not None:
        v0, from_ = r.version, r.commit
        if v1 == v0:
            raise ValueError(f"refusing to release same version: {v0} -> {v1}")
        elif v1 < v0:
            raise ValueError(f"refusing to downgrade version: {v0} -> {v1}")

    return Range(to, v1, from_, v0)

def generate_release_message(ctx: Context, range_: Range) -> str:
    (to, v1, from_, _) = astuple(range_)
    base_url = f"https://github.com/{ctx.github_repo.full_name}"
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

    return msg

@dataclass
class Prep:
    tag_name: str
    tag_message: str
    release_name: str
    release_message: str
    object: str
    type: str
    prerelease: bool

def prepare(args, ctx) -> Prep:
    range_ = prepare_range(ctx)
    if range_ is None:
        return

    (to, v1, from_, v0) = astuple(range_)
    logger.info("version: %s -> %s", v0, v1)
    logger.info("commits: %s..%s", from_ or "", to)

    if args.action == "prepare":
        dot_release = args.dot_release
        if not os.path.isabs(dot_release):
            dot_release = os.path.join(ctx.repo.working_dir, dot_release)

        if not args.dry_run:
            logger.info("writing .version-formatted release version to: %s", dot_release)
            with open(dot_release, "w") as f:
                emit_dot_version(v1, f)
        else:
            logger.info("would have written .version-formatted release version to: %s", dot_release)

    prep = Prep(
        tag_name = TAG_PREFIX + str(v1),
        tag_message = "",
        release_name = str(v1),
        release_message = generate_release_message(ctx, range_),
        object = to.hexsha,
        type = to.type,
        prerelease = bool(v1.prerelease)
    )

    if args.action == "prepare":
        dot_release_json = args.dot_release_json
        if not os.path.isabs(dot_release_json):
            dot_release_json = os.path.join(ctx.repo.working_dir, dot_release_json)

        if not args.dry_run:
            logger.info("writing release prep to: %s", dot_release_json)
            with open(dot_release_json, "w") as f:
                json.dump(asdict(prep), f)
        else:
            logger.info("would have written release prep to: %s", dot_release_json)

    return prep

def release(args_, ctx, prep):
    args = list(asdict(prep).values())[:-1]
    kwargs = { "prerelease": prep.prerelease }
    if not args_.dry_run:
        logger.info("creating release %s: %s", prep.tag_name, prep.release_name)
        ctx.github_repo.create_git_tag_and_release(*args, **kwargs)
    else:
        f = "%s.%s.%s" % (ctx.github_repo.__module__, ctx.github_repo.__class__.__qualname__, ctx.github_repo.create_git_tag_and_release.__name__)
        print(f"{f}(*{args}, **{kwargs})")

def run(args, ctx):
    if args.action == "release-prep":
        dot_release_json = args.dot_release_json
        if not os.path.isabs(dot_release_json):
            dot_release_json = os.path.join(ctx.repo.working_dir, dot_release_json)
        logger.info("reading release prep from: %s", dot_release_json)
        with open(dot_release_json) as f:
            prep = Prep(**json.load(f))
    else:
        prep = prepare(args, ctx)

    logger.debug("release prep: %s", prep)

    if args.action == "prepare":
        logger.debug("preparation done: bye")
        return

    release(args, ctx, prep)

def main():
    args = parse_args()
    setup_logger(args.log.upper())
    logger.debug(f"args: {args}")

    with closing(Context(args)) as ctx:
        run(args, ctx)

if __name__ == "__main__":
    main()
