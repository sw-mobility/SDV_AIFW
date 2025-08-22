let uid = localStorage.getItem('uid');
if (!uid || uid === 'undefined') {
  uid = '0001';
}
export { uid };