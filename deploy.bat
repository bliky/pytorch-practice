@echo off
rd /s /q _book
:: gitbook install
gitbook build
cd _book
git init
git add -A
git commit -m 'update book'
git push -f https://github.com/TubatuBD/pytorch-practice.git master:gh-pages