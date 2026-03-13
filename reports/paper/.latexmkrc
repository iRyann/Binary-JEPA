$out_dir = 'build';
$aux_dir = 'build/.aux';

# PDF mode
$pdf_mode = 1;

# BibTeX
$bibtex_use = 1;

# Nettoyage
push @clean_ext, qw(synctex.gz bbl run.xml bcf);
push @clean_full_ext, qw(
  aux fls fdb_latexmk log toc lof lot out nav snm
  idx ilg ind glg glo gls acr acn alg ist xdy
  vrb td0 td1 td2 td3 thm auxlock
  blg bcf xml
);

# Compilation (shell-escape si minted nécessaire)
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# https://tex.stackexchange.com/a/541990
# License: CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>
add_cus_dep( 'acn', 'acr', 0, 'makeglossaries' );
add_cus_dep( 'glo', 'gls', 0, 'makeglossaries' );
$clean_ext .= " acr acn alg glo gls glg";
sub makeglossaries {
     my ($base_name, $path) = fileparse( $_[0] );
     my @args = ( "-d", $path, $base_name );
     if ($silent) { unshift @args, "-q"; }
     return system "makeglossaries", @args;
}

# Qualité de vie
$preview_continuous_mode = 1;
$recorder = 1;
$emulate_aux_dir = 1;
