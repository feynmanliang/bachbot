#!/usr/bin/env python

import click



# import modules used here -- sys is a very standard one
import sys, click, logging


@click.command()
@click.option('--as-cowboy', '-c', is_flag=True, help='Greet as a cowboy.')
@click.argument('name', default='world', required=False)
def main(as_cowboy, name):
  """{{ cookiecutter.project_short_description }}"""

  # Setup logging
  if args.verbose:
    loglevel = logging.DEBUG
  else:
    loglevel = logging.INFO
  logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)

  greet = 'Howdy' if as_cowboy else 'Hello'
  click.echo('{0}, {1}.'.format(greet, name))


  # TODO Replace this with your actual code.
  print "Hello there."
  logging.info("You passed an argument.")
  logging.debug("Your Argument: %s" % args.argument)

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

# vim:set et sw=2 ts=8:
