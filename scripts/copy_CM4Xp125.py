#!/usr/bin/env python
# coding: utf-8

from copy_to_collab import *

import sys
import argparse
# defined command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--ppnames",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=key_vars.keys(),  # default if nothing is provided
)

CLI.add_argument(
  "--dmget",
  action="store_true",
  help="Issue dmget command before trying to copy the files"
)

CLI.add_argument(
  "--gcp",
  action="store_true",
  help="Issue dmget command before trying to copy the files"
)

# parse the command line
ppname_list = CLI.parse_args().ppnames
dmget = CLI.parse_args().dmget
gcp = CLI.parse_args().gcp

copy_CM4Xp125_to_collab(ppname_list, dmget=dmget, gcp=gcp)
