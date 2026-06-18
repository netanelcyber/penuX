/* kvmweb main.js */

const spinner = document.getElementById('spinner-overlay');
const toastContainer = document.getElementById('toast-container');

function showSpinner(msg) {
  if (spinner) {
    spinner.querySelector('span').textContent = msg || 'Working…';
    spinner.classList.add('active');
  }
}
function hideSpinner() {
  if (spinner) spinner.classList.remove('active');
}

function showToast(msg, type) {
  const t = document.createElement('div');
  t.className = 'toast toast-' + (type || 'success');
  t.textContent = msg;
  toastContainer.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function vmAction(vmName, action, callback) {
  const labels = {
    start: 'Starting…', shutdown: 'Shutting down…', destroy: 'Forcing off…',
    reboot: 'Rebooting…', suspend: 'Suspending…', resume: 'Resuming…',
    undefine: 'Deleting…', autostart_on: 'Enabling autostart…', autostart_off: 'Disabling autostart…'
  };
  showSpinner(labels[action] || 'Working…');

  fetch('/vm/' + encodeURIComponent(vmName) + '/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: action })
  })
  .then(r => r.json())
  .then(data => {
    hideSpinner();
    if (data.ok) {
      showToast(data.message || 'Done.', 'success');
      if (callback) callback(data);
      else setTimeout(() => window.location.reload(), 800);
    } else {
      showToast(data.error || 'Action failed.', 'error');
    }
  })
  .catch(err => {
    hideSpinner();
    showToast('Request failed: ' + err.message, 'error');
  });
}

function refreshStats() {
  fetch('/api/stats')
    .then(r => r.json())
    .then(data => {
      // Update state badges
      if (data.domains) {
        data.domains.forEach(vm => {
          const badge = document.querySelector('.state-badge[data-vm="' + vm.name + '"]');
          if (badge) {
            badge.textContent = vm.state;
            badge.className = 'badge badge-' + vm.state_class + ' state-badge';
            badge.dataset.vm = vm.name;
          }
        });
      }
      // Update host stat cards
      if (data.hinfo) {
        const h = data.hinfo;
        const memFree = document.getElementById('stat-mem-free');
        const memBar  = document.getElementById('stat-mem-bar');
        const vmActive = document.getElementById('stat-vm-active');
        const vmInactive = document.getElementById('stat-vm-inactive');
        if (memFree) memFree.textContent = h.free_mb + ' MB free';
        if (memBar)  memBar.style.width = Math.round((h.mem_mb - h.free_mb) / h.mem_mb * 100) + '%';
        if (vmActive)   vmActive.textContent   = h.active_vms   + ' running';
        if (vmInactive) vmInactive.textContent = h.inactive_vms + ' stopped';
      }
    })
    .catch(() => {});
}

// Wire up dashboard action buttons
document.querySelectorAll('.vm-action').forEach(btn => {
  btn.addEventListener('click', function() {
    const confirmMsg = this.dataset.confirm;
    if (confirmMsg && !confirm(confirmMsg)) return;
    vmAction(this.dataset.vm, this.dataset.action);
  });
});
