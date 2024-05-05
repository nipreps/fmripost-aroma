#!/bin/bash

echo Running tests

source .maint/ci/activate.sh

set -eu

# Required variables
echo CHECK_TYPE = $CHECK_TYPE

set -x

if [ "${CHECK_TYPE}" == "doc" ]; then
    cd doc
    make html && make doctest
elif [ "${CHECK_TYPE}" == "tests" ]; then
    pytest --doctest-modules --cov fmripost_aroma --cov-report xml \
        --junitxml=test-results.xml -v fmripost_aroma
else
    false
fi

set +eux

echo Done running tests
