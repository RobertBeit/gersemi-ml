const { randomUUID } = require("crypto");

const store = new Map();

const createRemoteRef = (value, type = null, snapshot = {}) => {
  const id = randomUUID();
  store.set(id, value);

  return {
    __remoteRef: id,
    __remoteType: type || value?.constructor?.name || "Object",
    __snapshot: snapshot,
  };
};

const getRemoteObject = (id) => store.get(id);

module.exports = {
  createRemoteRef,
  getRemoteObject,
};