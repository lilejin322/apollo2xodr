"""
Command line: ``apollo2xodr MAP [-o OUT]``.
"""

import argparse
import logging
import sys
import time
from . import FRAMES, __version__, convert
from .geometry import FIT_TOLERANCE

def _convert_parser():
    """
    Parse the command line arguments.
    """
    p = argparse.ArgumentParser(prog='apollo2xodr', description='Convert an Apollo HD map to OpenDRIVE 1.4.')
    p.add_argument('map', help='Apollo map: base_map.bin (binary) or a .txt text-format map')
    p.add_argument('-o', '--output', help='output .xodr (default: <map folder>/<folder name>.xodr)')
    p.add_argument('--frame', choices=FRAMES, default='lgsvl',
                   help='output coordinates: lgsvl = LGSVL/Scenic scene frame (default), utm = plain UTM')
    p.add_argument('--tolerance', type=float, default=FIT_TOLERANCE, metavar='M',
                   help=f'largest distance of the written reference lines from the Apollo polylines in metres '
                        f'(default {FIT_TOLERANCE})')
    p.add_argument('-v', '--verbose', action='store_true', help='print warnings about the map')
    p.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    return p

def main(argv=None) -> int:
    """
    Main function to convert the Apollo map to an OpenDRIVE format.

    :returns: 0 if successful
    :rtype: int
    """
    args = _convert_parser().parse_args(argv)     # Parse the command line arguments
    # Set up logging level, if --verbose, print INFO, otherwise ERROR
    logging.basicConfig(format='%(levelname)s %(message)s', level=logging.INFO if args.verbose else logging.ERROR)
    # conversion start time
    start = time.time()
    output = convert(args.map, args.output, frame=args.frame, tolerance=args.tolerance)
    print(f'Wrote {output} ({time.time() - start:.1f} s)')  # get the conversion time consumption
    return 0

if __name__ == '__main__':

    sys.exit(main())
