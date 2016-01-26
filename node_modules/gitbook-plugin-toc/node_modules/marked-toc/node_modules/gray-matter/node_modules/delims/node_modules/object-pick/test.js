/*!
 * object-pick <https://github.com/jonschlinkert/object-pick>
 *
 * Copyright (c) 2014 Jon Schlinkert, contributors.
 * Licensed under the MIT License
 */

'use strict';

var should = require('should');
var pick = require('./');

describe('.pick()', function () {
  it('should pick keys specified as strings.', function () {
    pick({a: 'a', b: 'b'}, 'a').should.eql({a: 'a'});
  });

  it('should pick keys specified as arrays.', function () {
    pick({a: 'a', b: 'b', c: 'c'}, ['a', 'b']).should.eql({a: 'a', b: 'b'});
    pick({foo: 'foo', bar: 'bar', baz: 'baz'}, ['foo', 'bar']).should.eql({foo: 'foo', bar: 'bar'});
  });

  it('should ignore keys that don\'t exist.', function () {
    pick({a: 'a', b: 'b', c: 'c'}, ['a', 'b', 'foo']).should.eql({a: 'a', b: 'b'});
    pick({foo: 'foo', bar: 'bar', baz: 'baz'}, ['foo', 'bar', 'abc']).should.eql({foo: 'foo', bar: 'bar'});
  });
});
