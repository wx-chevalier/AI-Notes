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

describe('fs', function () {
  it('should return true the path exists', function () {
    file.exists('test/fixtures/test.txt').should.be.true;
  });

  it('should return false when path does not exist', function () {
    file.exists('./some/random/file.json').should.be.false;
    file.exists('.', 'some', 'random', 'file.json').should.be.false;
  });

  it('should return true if a path is a real file', function() {
    file.isFile('package.json').should.be.true;
    file.isFile('README.md').should.be.true;
  });

  it('should return false if a path is not a real file', function() {
    file.isFile('test').should.be.false;
  });

  it('should return true if a path is a real directory', function() {
    file.isDir('test').should.be.true;
  });

  it('should return false if a path is not a real directory', function() {
    file.isDir('package.json').should.be.false;
    file.isDir('README.md').should.be.false;
  });

  it('should read the file', function () {
    file.readFileSync('test/fixtures/test.txt').should.eql('FILE CONTENTS!!!');
  });

  it('should read JSON', function () {
    file.readJSONSync('test/fixtures/test.json').should.eql({foo: {bar: "baz"} });
  });

  it('should read YAML', function () {
    file.readYAMLSync('test/fixtures/test.yaml').should.eql({foo: {bar: "baz"}});
  });

  it('should read detect JSON extension automatically', function () {
    file.readDataSync('test/fixtures/test.json').should.eql({foo: {bar: "baz"}});
  });

  it('should read YAML automatically', function () {
    file.readDataSync('test/fixtures/test.yaml').should.eql({foo: {bar: "baz"}});
  });

  it('should make a directory, synchronously', function() {
    file.mkdirSync('test/actual/new/folder/sync');
    file.exists('test/actual/new/folder/sync').should.be.true;
  });

  it('should remove a directory, synchronously', function() {
    file.del('test/actual/new/folder/sync');
    file.exists('test/actual/new/folder/sync').should.be.false;
  });

  it('should "delete" a directory, synchronously', function() {
    file.mkdirSync('test/actual/new/folder/sync');
    file.del('test/actual/new/folder/sync');
    file.exists('test/actual/new/folder/sync').should.be.false;
  });

  it('should write a file', function () {
    file.writeFileSync('test/actual/test.txt', 'FILE CONTENTS!!!');
    file.readFileSync('test/actual/test.txt').should.eql('FILE CONTENTS!!!');
    file.del('test/actual/test.txt');
  });

  it('should write JSON', function () {
    file.writeJSONSync('test/actual/test.json', {foo: {bar: "baz"}});
    file.readJSONSync('test/actual/test.json').should.eql({foo: {bar: "baz"}});
    file.del('test/actual/test.json');
  });

  it('should write YAML', function () {
    file.writeYAMLSync('test/actual/test.yaml', {foo: {bar: "baz"}});
    file.readYAMLSync('test/actual/test.yaml').should.eql({foo: {bar: "baz"}});
    file.del('test/actual/test.yaml');
  });

  it('should write JSON', function () {
    file.writeDataSync('test/actual/test.json', {foo: {bar: "baz"}});
    file.readDataSync('test/actual/test.json').should.eql({foo: {bar: "baz"}});
    file.del('test/actual/test.json');
  });

  it('should write YAML', function () {
    file.writeDataSync('test/actual/test.yaml', {foo: {bar: "baz"}});
    file.readDataSync('test/actual/test.yaml').should.eql({foo: {bar: "baz"}});
    file.del('test/actual/test.yaml');
  });

  describe('glob', function () {
    it('should return an array of files.', function () {
      var files = file.glob.sync('test/**/*.js');
      files.should.be.an.array;
      (files.length > 2).should.be.true;
    });
  });
});