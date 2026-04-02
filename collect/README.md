## Mail / Outlook wget import

If the NRAO archive emails arrive in macOS Mail or Outlook, you can fill blank
`wget_command` cells in `small_subset/small_selection.csv` automatically:

```bash
python3 /Users/u1528314/repos/radioastro-ml/collect/import_outlook_wget.py --dry-run
python3 /Users/u1528314/repos/radioastro-ml/collect/import_outlook_wget.py
```

What it does:

- scans recent Mail/Outlook messages for NRAO `wget` commands
- matches each message to a pending CSV row using the calibrator name/folder or
  the NRAO `productViewer` / SDM id in the email body
- writes the command into the existing `wget_command` column so
  `scripts/extraction_pipeline.py` can keep working as-is

Default backend:

- macOS Mail (`--backend mail`) is the default and recommended path
- Outlook is still available with `--backend outlook`

Useful flags:

- `--days-back 30` to scan older mail
- `--mail-account Google` to target a specific Mail app account
- `--mailbox INBOX` to target a specific Mail app mailbox
- `--sender do-not-reply@nrao.edu` to restrict Mail app scans to NRAO mail
- all Outlook folders are scanned by default; use `--inbox-only` to restrict it
- `--unread-only` to limit the scan to unread mail
- `--messages-json some_file.json` to test against exported mail without live Outlook access

Expected JSON format for `--messages-json`:

```json
[
  {
    "message_id": "abc123",
    "subject": "Your NRAO data request is ready",
    "received_at": "2026-04-02 10:00:00",
    "content": "wget ... https://dl-dsoc.nrao.edu/..."
  }
]
```

On first use, macOS may ask for Automation permission so `osascript` can read Outlook.
