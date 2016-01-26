var log = require('../');

/**
 * TODO: unit tests. this is a placeholder
 */

log.mode.verbose = true;

log.info('this is info.');
log.verbose.info('this is verbose info.');
log.verbose.info('this is more verbose info.');


log.mode.verbose = false;


log.magenta('foo');

log('This is a message');
log.bold('This is a bold message');
log.info('This is an info message');
log.success('This is a success message');
log.warn('This is a warning message');
log.error('This is a error message');



log.bold('>> This is', 'a bold message');
log.info('>> This is', 'an info message');
log.success('>> This is', 'a success message');
log.warn('>> This is', 'a warning message');
log.error('>> This is', 'a error message');



log.verbose.bold('This is a bold verbose message');
log.verbose.info('This is an info verbose message');
log.verbose.success('This is a success verbose message');
log.verbose.warn('This is a warning verbose message');
log.verbose.error('This is a error verbose message');

log.mode.verbose = true;


log.verbose.bold('This is a bold verbose message');
log.verbose.info('This is an info verbose message');
log.verbose.success('This is a success verbose message');
log.verbose.warn('This is a warning verbose message');
log.verbose.error('This is a error verbose message');

// Fatal messages last
log.verbose.fatal('This is a fatal verbose message');
