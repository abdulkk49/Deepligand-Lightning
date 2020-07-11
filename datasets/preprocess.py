from utils import *
import argparse, shutil
from os import makedirs
from os.path import exists, realpath, join, dirname

def parse_args():
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("-a", "--action", required=True)
    return parser.parse_args()

pwd = dirname(realpath("__file__"))
args = parse_args()
args.file = realpath(args.file)
args.outdir = realpath(args.outdir)
dt = args.action

# if not exists(args.outdir):
#     create_dir(args.outdir)

# actionDir = join(pwd,dt)

# baseAction = join(actionDir, "CV")

# mergefiles(baseAction, "_" + dt) 
# createraw(actionDir, dt)
# createlrm(actionDir, dt)

# # Load pseudo-sequences
# pseudo_seq_file = join(pwd, 'data/MHC_pseudo.dat')
# pseudo_seq_dict = dict()
# with open(pseudo_seq_file) as f:
#     for x in f:
#         line = x.split()
#         pseudo_seq_dict[line[0]] = line[1]

# # Map MHC names
# print('MHC mapping..')
# mhc_mapper(args.file, args.outdir, pseudo_seq_dict, dt)

# # Pad peptides to 40 AA
# print('Peptide padding')
# padseq(join(args.outdir, dt + '.pep'), '.pep', pad2len = {'.pep':40})

# # Tokenize
# print('Tokenizing')
# tokenize(join(args.outdir, dt + '.pep'), join(args.outdir, dt + '.pep.token'))

# # Peptide embedding
# print('Peptide embedding')
# system(' '.join(['python {}/datasets/elmo_embed.py -d {} -e {} -t {} --trial_num -1'.format(pwd, args.outdir, join(pwd, 'data'), dt)]))

# Embed
print('Data embedding')
embed(args.outdir, args.outdir, dt)