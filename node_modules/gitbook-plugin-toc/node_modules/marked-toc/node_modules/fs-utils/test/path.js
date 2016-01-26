/**
 * fs-utils <https://github.com/assemble/fs-utils>
 *
 * Copyright (c) 2014 Jon Schlinkert, Brian Woodward, contributors.
 * Licensed under the MIT license.
 */
'use strict';

var should = require('should');
var path = require('path');
var file = require('..');

// Normalize slashes in some test results
var normalize = file.forwardSlash;

describe('Normalize slashes', function() {
  it('should normalize slash', function() {
    file.forwardSlash('foo\\bar/baz').should.equal('foo/bar/baz');
  });
});

describe('Trailing slashes', function() {
  describe('Add trailing slashes', function() {
    it('should add a trailing slash when it appears to be a directory', function() {
      normalize(file.addSlash('foo/bar/baz')).should.equal('foo/bar/baz/');
      normalize(file.addSlash('/foo/bar/baz')).should.equal('/foo/bar/baz/');
      normalize(file.addSlash('./foo/bar.baz/quux')).should.equal('./foo/bar.baz/quux/');
      normalize(file.addSlash('./foo/bar/baz')).should.equal('./foo/bar/baz/');
      normalize(file.addSlash('\\foo\\bar\\baz')).should.equal('/foo/bar/baz/');
      normalize(file.addSlash('foo\\bar\\baz')).should.equal('foo/bar/baz/');
    });

    it('should not add a trailing slash when it already has one', function() {
      normalize(file.addSlash('foo/bar/baz/')).should.equal('foo/bar/baz/');
      normalize(file.addSlash('/foo/bar/baz/')).should.equal('/foo/bar/baz/');
    });

    it('should not add a trailing slash when it appears to be a file', function() {
      normalize(file.addSlash('./foo/bar/baz.md')).should.equal('./foo/bar/baz.md');
      normalize(file.addSlash('/foo/bar/baz.md')).should.equal('/foo/bar/baz.md');
      normalize(file.addSlash('\\foo\\bar\\baz.md')).should.equal('/foo/bar/baz.md');
    });
  });

  describe('Remove trailing slashes', function() {
    it('should remove a trailing slash from the given file path', function() {
      normalize(file.removeSlash('./one/two/')).should.equal('./one/two');
      normalize(file.removeSlash('/three/four/five/')).should.equal('/three/four/five');
      normalize(file.removeSlash('\\six\\seven\\eight\\')).should.equal('/six/seven/eight');
    });
  });
});

describe('lastChar:', function() {
  it('should return the last character in the given file path', function() {
    file.lastChar('foo/bar/baz/quux/file.ext').should.equal('t');
    file.lastChar('foo/bar/baz/quux').should.equal('x');
    file.lastChar('foo/bar/baz/quux/').should.equal('/');
  });
});

