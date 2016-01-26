var toc = require('marked-toc');

module.exports = {
  book: {
  },
  hooks: {
    "page:before": function(page) {
      var tmpl = '<%= depth %><%= bullet %>[<%= heading %>](#<%= url %>)\n';
      page.content = toc.insert(page.content, {template: tmpl});
      return page;
    }
  }
};
