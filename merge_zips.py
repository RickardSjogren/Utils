""" Merge given zip-archives into single archive. """
import argparse
import zipfile
import glob


def merge_zip_files(zip_files, output, verbosity=0,
                    allow_zip64=False, mode=zipfile.ZIP_DEFLATED):
    """ Merge contents from `zip_files` into `output`.

    Parameters
    ----------
    zip_files : list[str]
        Paths to source zip-files.
    output : str
        Path to output archive.
    verbosity : int
        Verbosity level.
    allow_zip64 : bool
        Allow zip64 compression, default False.
    mode : {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        Compression mode, default `zipfile.ZIP_DEFLATED`.
    """
    with zipfile.ZipFile(output, 'a',
                         compression=mode,
                         allowZip64=allow_zip64) as dst:
        for path in zip_files:

            if not zipfile.is_zipfile(path):
                print_if_verbose('Skips: {}'.format(path), verbosity)
                continue

            print_if_verbose('Merges {} to destination.'.format(path),
                             verbosity)
            with zipfile.ZipFile(path) as src:
                for name in src.namelist():
                    msg = 'Copies {} from {} to destination'.format(name, path)
                    print_if_verbose(msg, verbosity, 2)
                    dst.writestr(name, src.open(name).read())


def print_if_verbose(out, verbosity, min_count=1):
    if verbosity >= min_count:
        print(out)


def make_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('src', help='Path to files to be merged; '
                                    'accepts * as wildcard for directories '
                                    'or filenames.')
    default_output = 'merged.zip'
    parser.add_argument('-o', '--output',
                        help='Output path (default {})'.format(default_output),
                        default=default_output)
    parser.add_argument('--no_compression', action='store_false',
                        help='If set, do not compress output.')
    parser.add_argument('--zip64', action='store_true', default=False,
                        help='If set, allow Zip64.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase output verbosity.')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    if args.no_compression:
        mode = zipfile.ZIP_STORED
    else:
        mode = zipfile.ZIP_DEFLATED

    merge_zip_files(glob.glob(args.src),
                    args.output,
                    args.verbose,
                    args.zip64,
                    mode)