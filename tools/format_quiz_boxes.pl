use strict;
use warnings;

for my $path (@ARGV) {
    open my $in, '<:encoding(UTF-8)', $path or die "$path: $!";
    local $/;
    my $text = <$in>;
    close $in;

    $text =~ s/^## (\d+)\. ([^\n]+)\n\n/:::{admonition} Quiz $1 — $2\n:class: note\n\n/gm;
    $text =~ s/\n:::\{admonition\} Solution/\n:::\n\n:::{admonition} Solution/g;

    open my $out, '>:encoding(UTF-8)', $path or die "$path: $!";
    print {$out} $text;
    close $out;
}
