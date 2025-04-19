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

def mk_github(args):
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None:
        auth = github.Auth.Token(token)
    else:
        auth = None
    return github.Github(auth=auth)

def get_release_to_tag(args):
    repo = mk_github(args).get_repo(args.github_repo)

    logger.info("fetching releases from: %s", repo.full_name)

    releases = {}
    for r in repo.get_releases():
        if r.draft:
            continue

        releases[r.tag_name] = {
            "tag_name": r.tag_name,
            "name": r.name,
            "prerelease": r.prerelease,
        }
    return releases

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

def sandbox(args):
    if args.local_repo is None:
        raise NotImplementedError("clone repo")
    repo = git.Repo(args.local_repo)

    releases = pickle_expr("releases", lambda: get_release_to_tag(args))

    # TODO ~> class Releases
    c2r = {}
    for tag_name, r in releases.items():
        c = repo.tag(tag_name).commit
        logger.debug("resolved: tag %s -> %s", tag_name, c.hexsha)
        r["commit"] = c
        c2r[c] = r

    target = repo.commit(args.rev)
    logger.info("aiming to release: %s", target)

    ptr, r = target, c2r.get(target)
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
    if r is None:
        logger.info("no previous release")
    else:
        logger.info("previous release: %s", r)

    def dot_version(commit):
        try:
            return semver.parse(commit.tree[".version"].data_stream.read().decode())
        except KeyError:
            return None

    v1 = dot_version(target)
    if r is not None:
        v0 = dot_version(r["commit"])
    else:
        v0 = None

    print(v1, r, v0)

    # match (v1, r, v0):

def main():
    args = parse_args()
    setup_logger(args.log.upper())
    logger.debug(f"args: {args}")

    sandbox(args)

if __name__ == "__main__":
    main()
