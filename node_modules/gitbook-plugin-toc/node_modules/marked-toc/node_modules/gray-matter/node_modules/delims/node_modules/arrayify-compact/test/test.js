/*!
 * arrayify-compact <https://github.com/jonschlinkert/arrayify-compact>
 *
 * Copyright (c) 2014 Jon Schlinkert, contributors.
 * Licensed under the MIT License
 */

'use strict';

var assert = require('assert');
var arrayify = require('../');

describe('arrayify', function () {
  it('should arrayify the value', function () {
    var actual = arrayify('a');
    assert.deepEqual(actual, ['a']);
  });

  it('should flatten the arrays', function () {
    var actual = arrayify(['a', 'b', ['c', ['d']]]);
    assert.deepEqual(actual, ['a', 'b', 'c', 'd']);
  });

  it('should compact the arrays, by removing all falsey values', function () {
    var actual = arrayify(['a', 'b', ['c', ['d'], null, false, 0, NaN, '', [], undefined]]);
    assert.deepEqual(actual, ['a', 'b', 'c', 'd']);
  });
});